import logging
from uuid import uuid4
from dataclasses import dataclass
from typing import TypeVar
from tqdm import tqdm

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    BaseMessage
)
from langgraph.graph.state import CompiledStateGraph

from fasta2a.worker import Worker
from fasta2a.schema import (
    Artifact,
    TaskSendParams,
    TaskIdParams,
    Message,
    Part,
    TextPart
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# AgentWorker output type needs to be invariant for use in both parameter and return positions
WorkerOutputT = TypeVar('WorkerOutputT')

@dataclass
class LangGraphAgentWorker(Worker):
    agent: CompiledStateGraph
    
    async def run_task(self, params: TaskSendParams) -> None:
        task = await self.storage.load_task(params['id'])
        if task is None:
            raise ValueError(f'Task {params["id"]} not found')
        
        if task['status']['state'] != 'submitted':
            raise ValueError(  # pragma: no cover
                f'Task {params["id"]} has already been processed (state: {task["status"]["state"]})'
            )
        
        await self.storage.update_task(task['id'], state='working')
        
        message_history = await self.storage.load_context(task['context_id']) or []
        message_history.extend(self.build_message_history(task.get('history', [])))
        
        ## Invoke LangGraph Agent
        try:
            result = await self.agent.invoke(
                {'messages': message_history}
            )
            response = result['messages'][-1].content
            await self.storage.update_context(
                task['context_id'],
                result['messages'] #dump all messages into the result
            )
            
            # Convert new messages to A2A format for task history.  Note: we only keep human and agent messages
            a2a_messages: list[Message] = self._response_to_a2a_message(result['messages'])
            
            # Build artifacts
            artifacts = self.build_artifacts(response)
            
        except Exception as e:
            await self.storage.update_task(task['id'], state='failed')
            raise ValueError(e)
        
        else:
            await self.storage.update_task(
                task['id'],
                state='completed',
                new_artifacts=artifacts,
                new_messages=a2a_messages
            )
    
    def build_artifacts(self, result: WorkerOutputT) -> list[Artifact]:
        """Builds artifacts from the agent's results.
        
        All agent outputs become artifacts to mark them as durable task outputs.
        For string results, we use TextPart.
        """
        artifact_id = str(uuid4())
        part = TextPart(text=result, kind='text')
        return [Artifact(artifact_id=artifact_id, parts=[part], name='result')]
    
    async def cancel_task(self, params: TaskIdParams) -> None:
        raise NotImplementedError(
            'LangGraphAgentWorker does not support task cancellation yet'
        )
    
    def build_message_history(self, history: list[Message]) -> list[BaseMessage]:
        """Assembles LangChain message history from A2A messages."""
        messages = []
        for message in history:
            content = self._get_content_from_a2a_part(message['parts'])
            if message.role == 'user':
                messages.append(HumanMessage(content=content))
        return messages
    
    def _get_content_from_a2a_part(self, parts: list[Part]) -> list[BaseMessage]:
        """
        Converts A2A Part objects into LangChain Message objects.
        
        Should only contain one part, and it should always be a string.
        """
        model_parts = [{'text': part['text']} for part in parts if isinstance(part, TextPart)]
        assert len(model_parts) == 1, "Expected exactly one text part"
        return model_parts[0]['text']
    
    def _response_to_a2a_message(self, messages: list[BaseMessage]):
        """Convert LangChain messages to A2A Message format."""
        results = []
        for message in tqdm(messages):
            part = TextPart(text=message.content, kind='text')
            if isinstance(message, HumanMessage):
                role = 'user'
            elif isinstance(message, AIMessage):
                role = 'agent'
            results.append(
                    Message(
                        role=role,
                        parts = [part],
                        kind='message',
                        message_id=str(uuid4())
                    )
                )
        return results
        
        