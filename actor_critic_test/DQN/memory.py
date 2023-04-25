import torch
import random 
from typing import Tuple, List

class MemoryBuffer():

    def __init__(self, buff_size: int, state_size: int, batch_size: int):
        """
            Params:
            buff_size: number of items to store in buffer before replacing items
            state_size: number of elements in the state vector

        """

        self.buff_ctr   : int = 0
        self.batch_size : int = batch_size
        self.buff_size  : int = buff_size

        self.state_buff  : torch.Tensor =  torch.zeros((buff_size, state_size), dtype=torch.float32)
        self.state_buff2 : torch.Tensor =  torch.zeros((buff_size, state_size), dtype=torch.float32)
        self.term_buff   : torch.Tensor =  torch.zeros((buff_size)            , dtype=torch.int8   )
        self.action_buff : torch.Tensor =  torch.zeros((buff_size)            , dtype=torch.int8   )
        self.reward_buff : torch.Tensor =  torch.zeros((buff_size)            , dtype=torch.int8   )
        

    def getBatch(self) -> List[Tuple]:
        """
            Returns:
            List[Tuple( S, a, S', r)]
            Returns a list of size (batch_size)
            which contains a tuple of state, action,
            next_state, reward, terminal_flag
        """
        batch_size: int = min(self.batch_size, self.buff_ctr)

        indices: List[int] = [ random.randint(0, min(self.buff_ctr,self.buff_size) - 1) \
                               for _ in range(batch_size) ]
        batch : List[Tuple] = []


        for ind in indices:
            s = self.state_buff[ind]
            a = self.action_buff[ind]
            s_= self.state_buff2[ind]
            r = self.reward_buff[ind]
            T = self.term_buff[ind]
            batch.append( (s, a, s_, r, T) )


        return batch

    
    def addMemory(self, state: torch.Tensor, action: int, state2: torch.Tensor, reward: int, terminal: int) -> None:
        """
            Params:
            state: representation of current state
            action: representation of action taken
            state2: representation of transitioned state
            reward: represetation of reward recieved
            terminal: flag representing of episode ended here

        """
        ind: int = self.buff_ctr % self.buff_size

        self.state_buff[ind]  = state
        self.state_buff2[ind] = state2
        self.term_buff[ind] = terminal
        self.action_buff[ind] = action
        self.reward_buff[ind] = reward

        self.buff_ctr+= 1 

