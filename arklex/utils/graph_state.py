from typing import TypedDict, Any, Annotated, Optional, Union, List, Dict
import janus
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
import uuid
from queue import Queue


### Bot-related classes
class BotConfig(BaseModel):
    bot_id: str
    version: str
    language: str
    bot_type: str

### Message-related classes

class ConvoMessage(BaseModel):
    history: str # it could be the whole original message or the summarization of the previous conversation from memory module
    message: str


class OrchestratorMessage(BaseModel):
    message: str
    attribute: dict


### Slot-related classes

class Slot(BaseModel):
    name: str
    type: Union[str, int, float, bool, None]
    value: Union[str, int, float, bool, List[str], None]
    enum: Optional[list[Union[str, int, float, bool, None]]]
    description: str
    prompt: str
    required: bool
    verified: bool


class Slots(BaseModel):
    slots: list[Slot]

# May not be needed
class SlotDetail(Slot):
    verified_value: str
    confirmed: bool

class Verification(BaseModel):
    thought: str
    verification_needed: bool


### Task status-related classes

class StatusEnum(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    STAY = "stay"

class Timing(BaseModel):
    taskgraph: Optional[float] = None

class ResourceRecord(BaseModel):
    info: Dict
    input: List = Field(default_factory=list)
    output: str = Field(default="")
    steps: List = Field(default_factory=list)

class Metadata(BaseModel):
    # TODO: May need to initialize the metadata(i.e. chat_id, turn_id) based on the conversation database
    chat_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    turn_id: int = 0
    hitl: Optional[str] = Field(default=None)
    timing: Timing = Field(default_factory=Timing)

class MessageState(BaseModel):
    # system configuration
    sys_instruct: str
    # bot configuration 
    bot_config: BotConfig
    # input message
    user_message: ConvoMessage
    orchestrator_message: OrchestratorMessage
    # action trajectory
    function_calling_trajectory: List[Dict[str, Any]]
    trajectory: List[List[ResourceRecord]]
    # message flow between different nodes
    message_flow: str = Field(description="message flow between different nodes")
    # final response
    response: str = Field(default="")
    # task-related params
    status: StatusEnum = Field(default=StatusEnum.INCOMPLETE)
    slots: Dict[str, List[Slot]] = Field(description="record the dialogue states of each action")
    metadata: Metadata
    # stream
    is_stream: bool
    message_queue: Any = Field(default_factory=Queue, exclude=True)


class PathNode(BaseModel):
    node_id: str
    is_skipped: bool = False
    in_flow_stack: bool = False
    nested_graph_node_value: Optional[str] = None
    nested_graph_leaf_jump: Optional[str] = None


class Taskgraph(BaseModel):
    # Need add global intent
    dialog_states: Dict[str, List[Slot]] = Field(default_factory=dict)
    path: List[PathNode] = Field(default_factory=list)
    curr_node: str = Field(default="")
    intent: str = Field(default="")
    node_limit: Dict[str, int] = Field(default_factory=dict)
    nlu_records: List = Field(default_factory=list)
    node_status: Dict[str, StatusEnum] = Field(default_factory=dict)
    available_global_intents: List = Field(default_factory=list)


class Memory(BaseModel):
    trajectory: List[List[ResourceRecord]] = Field(default_factory=list)
    function_calling_trajectory: List[Dict[str, Any]] = Field(default_factory=list)
    
class Params(BaseModel):
    metadata: Metadata = Field(default_factory=Metadata)
    taskgraph: Taskgraph = Field(default_factory=Taskgraph)
    memory: Memory = Field(default_factory=Memory)

class NodeInfo(BaseModel):
    resource_id: str = Field(default="")
    resource_name: str = Field(default="")
    can_skipped: bool = Field(default=False)
    is_leaf: bool = Field(default=False)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    add_flow_stack: Optional[bool] = Field(default=False)

class OrchestratorResp(BaseModel):
    answer: str = Field(default="")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    human_in_the_loop: Optional[str] = Field(default=None)