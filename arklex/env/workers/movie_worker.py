import os
import requests
from langgraph.graph import StateGraph, START
from langchain_openai import ChatOpenAI

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.graph_state import MessageState
from arklex.env.tools.utils import ToolGenerator
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP
from arklex.utils.graph_state import StatusEnum


@register_worker
class MovieGraphWorker(BaseWorker):
    description = "Recommends movies using mood, genre, and exclusions. Robust to typos and incomplete input."

    def __init__(self):
        super().__init__()
        self.llm = PROVIDER_MAP.get(MODEL["llm_provider"], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000
        )
        self.graph = self._create_action_graph()

    def _create_action_graph(self):
        workflow = StateGraph(MessageState)

        workflow.add_node("parse_mood_genre", self.parse_mood_genre)
        workflow.add_node("fetch_movies", self.fetch_movies)
        workflow.add_node("fallback", self.fallback_handler)
        workflow.add_node("final_response", self.final_response)

        workflow.add_edge(START, "parse_mood_genre")
        workflow.add_conditional_edges(
            "parse_mood_genre", self.check_parse_success,
            {"success": "fetch_movies", "fail": "fallback"}
        )
        workflow.add_conditional_edges(
            "fetch_movies", self.check_fetch_success,
            {"success": "final_response", "fail": "fallback"}
        )
        workflow.add_edge("fallback", "final_response")

        return workflow

    def _execute(self, msg_state: MessageState):
        return self.graph.compile().invoke(msg_state)

    def final_response(self, state: MessageState):
        # Use the response from fetch_movies or fallback
        response = getattr(state, "response", None) or state.response
        state.response = response
        return state

    
    def parse_mood_genre(self, state: MessageState):
        try:
            history = list(state.message_queue.queue) if hasattr(state.message_queue, "queue") else []
            recent_context = " ".join(history[-5:] if history else [state.user_message.message])

            prompt = f"""
            The user is describing the type of movie they want. This could include mood, genre, or exclusions.

            Example input:
            "I want something romantic but not a musical."

            Conversation:
            "{recent_context}"

            Respond in this strict format:
            Mood: [e.g., happy, sad, romantic]
            Genre: [e.g., Comedy, Drama, Romance]
            Exclude: [list of exclusions or None]
            """

            result = self.llm.invoke(prompt).content.strip()
            mood = next(line for line in result.splitlines() if line.lower().startswith("mood:")).split(":", 1)[1].strip()
            genre = next(line for line in result.splitlines() if line.lower().startswith("genre:")).split(":", 1)[1].strip()
            exclude = next(line for line in result.splitlines() if line.lower().startswith("exclude:")).split(":", 1)[1].strip()

            # Store in metadata (only writable allowed place for dynamic info)
            state.metadata.__dict__["mood"] = mood
            state.metadata.__dict__["genres"] = [genre]
            state.metadata.__dict__["exclude"] = exclude
            state.status = StatusEnum.COMPLETE
            return state

        except Exception as e:
            state.status = StatusEnum.INCOMPLETE
            return state

    def fetch_movies(self, state: MessageState):
        try:
            api_key = os.getenv("TMDB_API_KEY")
            genres = state.metadata.__dict__.get("genres", ["Drama"])
            if isinstance(genres, str):
                genres = [genres]

            genre_url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={api_key}&language=en-US"
            response = requests.get(genre_url)
            response.raise_for_status()
            genre_map = {g["name"].lower(): g["id"] for g in response.json().get("genres", [])}
            genre_ids = [str(genre_map[g.lower()]) for g in genres if g.lower() in genre_map]

            if not genre_ids:
                state.status = StatusEnum.INCOMPLETE
                return state

            discover_url = (
                f"https://api.themoviedb.org/3/discover/movie?api_key={api_key}"
                f"&with_genres={','.join(genre_ids)}&sort_by=vote_average.desc&vote_count.gte=100"
            )
            movie_res = requests.get(discover_url)
            movie_res.raise_for_status()
            movies = movie_res.json().get("results", [])[:5]

            if not movies:
                state.status = StatusEnum.INCOMPLETE
                return state

            reply = "🎬 Based on your preferences, here are some movie picks:\n\n"
            for m in movies:
                title = m.get("title", "Unknown")
                overview = m.get("overview", "")[:100]
                rating = m.get("vote_average", "N/A")
                release = m.get("release_date", "Unknown")
                reply += f"• **{title}** ({release}) - ⭐ {rating}/10\n  {overview}...\n\n"

            state.status = StatusEnum.COMPLETE
            state.response = reply
            return state

        except Exception as e:
            state.status = StatusEnum.INCOMPLETE
            return state

    def fallback_handler(self, state: MessageState):
        message = (
            "😕 I got a bit confused. Could you try telling me again how you're feeling "
            "or what kind of movie you're in the mood for?"
        )
        state.response = message
        return state

    def check_parse_success(self, state: MessageState) -> str:
        return "success" if str(state.status) == "StatusEnum.COMPLETE" else "fail"

    def check_fetch_success(self, state: MessageState) -> str:
        return "success" if str(state.status) == "StatusEnum.COMPLETE" else "fail"
