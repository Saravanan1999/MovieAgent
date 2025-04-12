import os
from openai import OpenAI
import requests
from arklex.env.workers.worker import BaseWorker
from arklex.utils.graph_state import StatusEnum  # ✅ Added

client = OpenAI()

class MoodToGenreWorker(BaseWorker):
    description = "Maps user mood to TMDB genre names, even with typos."

    def _execute(self, msg_state):
        user_input = msg_state.message_queue[-1]

        prompt = f"""
        The user said: "{user_input}". It may contain spelling mistakes or casual phrasing.
        Based on this, what is the most likely mood they're expressing?

        Respond with only the mood (like 'happy', 'sad', 'relaxed', 'tired', 'romantic', 'excited', etc.).
        Then suggest a single movie genre that best matches the mood from this list:
        ['Comedy', 'Drama', 'Action', 'Romance', 'Horror', 'Thriller', 'Adventure', 'Animation', 'Sci-Fi'].

        Respond in this format (no explanation):
        Mood: [mood]
        Genre: [genre]
        """

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )

        content = completion.choices[0].message.content.strip()

        try:
            mood_line = next(line for line in content.split("\n") if line.lower().startswith("mood"))
            genre_line = next(line for line in content.split("\n") if line.lower().startswith("genre"))
            mood = mood_line.split(":", 1)[-1].strip()
            genre = genre_line.split(":", 1)[-1].strip()
        except Exception:
            return self.create_response(message="Sorry, I couldn't interpret your mood clearly. Could you rephrase?", data={})

        msg_state.metadata.taskgraph.node_status[msg_state.metadata.taskgraph.curr_node] = StatusEnum.COMPLETE.value

        return {
            "message_queue": msg_state.message_queue,
            "genre": genre,
            "mood": mood
        }



class MovieFetchWorker(BaseWorker):
    description = "Fetches popular movies by genre using TMDB API."

    def _execute(self, msg_state):
        api_key = os.getenv("TMDB_API_KEY")

        try:
            genres = msg_state.data.get("genres", ["Drama"])
        except AttributeError:
            # fallback for compatibility
            genres = getattr(msg_state, "genre", ["Drama"])
            if isinstance(genres, str):
                genres = [genres]

        try:
            genre_url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={api_key}&language=en-US"
            response = requests.get(genre_url)
            response.raise_for_status()
            genre_map = {g['name'].lower(): g['id'] for g in response.json().get("genres", [])}
            genre_ids = [str(genre_map[g.lower()]) for g in genres if g.lower() in genre_map]

            if not genre_ids:
                msg = "Sorry, I couldn't find a matching genre in TMDB 😢"
                return self._finalize_response(msg_state, msg)

            discover_url = (
                f"https://api.themoviedb.org/3/discover/movie?api_key={api_key}"
                f"&with_genres={','.join(genre_ids)}&sort_by=vote_average.desc&vote_count.gte=100"
            )
            movie_res = requests.get(discover_url)
            movie_res.raise_for_status()
            movies = movie_res.json().get("results", [])[:4]

            if not movies:
                msg = "No movies found matching your mood 😢"
                return self._finalize_response(msg_state, msg)

            reply = "🎥 Here are some movie picks for your mood:\n\n"
            for m in movies:
                title = m.get("title", "Unknown")
                rating = m.get("vote_average", "N/A")
                release = m.get("release_date", "Unknown")
                overview = m.get("overview", "")
                reply += f"• {title} ({release}) – ⭐ {rating}/10\n  {overview[:100]}...\n\n"

            return self._finalize_response(msg_state, reply)

        except Exception as e:
            return self._finalize_response(msg_state, f"Something went wrong fetching movie data 😕: {e}")

    def _finalize_response(self, msg_state, message):
        msg_state.metadata.taskgraph.node_status[
            msg_state.metadata.taskgraph.curr_node
        ] = StatusEnum.COMPLETE.value
        return self.create_response(message=message)