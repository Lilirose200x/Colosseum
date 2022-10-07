# Student agent: Add your own agent here
from agents.agent import Agent
from .random_agent import RandomAgent
from store import register_agent
import numpy as np
import traceback
from copy import deepcopy
import sys

from agents import *
from constants import *

import world 


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.autoplay = True
        self.random_agent = RandomAgent()
        self.search_steps = 3
        self.search_width = 40
        self.num_simulates = 100
        self.uct_c = 1
        self.fakeworld = world.World(board_size=6)
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def random_walk(self, chess_board, my_pos, adv_pos, max_step):
        # Moves (Up, Right, Down, Left)
        ori_pos = deepcopy(my_pos)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = np.random.randint(0, max_step + 1)

        # Random Walk
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos
        while chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)

        return my_pos, dir

    def compute_uct(self, q, n_v, n):
        return q/n_v+self.uct_c*np.sqrt(np.log(n)/n_v)


    def step(self, chess_board, my_pos, adv_pos, max_step):
        Q = {}
        for _ in range(self.search_width):
            pos, dir = self.random_walk(chess_board, my_pos, adv_pos, max_step) 
            r, c = pos
            uct = self.compute_uct(0, 1, self.num_simulates)
            max_score = 0
            for i in range(4):
                if chess_board[r, c, i]:
                    continue
                chess_board_clone = deepcopy(chess_board)
                self.fakeworld.turn = 0
                self.fakeworld.board_size = chess_board.shape[0]
                self.fakeworld.max_step = (chess_board.shape[0]+1)//2
                self.fakeworld.p1_pos = np.asarray(pos, dtype=int)
                self.fakeworld.p0_pos = np.asarray(adv_pos, dtype=int)
                self.fakeworld.chess_board = chess_board_clone
                r, c = pos
                self.fakeworld.set_barrier(r, c, i)
                is_end, p0_score, p1_score = self.fakeworld.check_endgame()
                if is_end and p1_score>p0_score:
                    return pos, i
                if max_score<p1_score/p0_score:
                    max_score = p1_score/p0_score
                    dir = i
            Q[str(r)+'-'+str(c)+'-'+str(dir)] = [1, 1, r, c, dir, uct]
        #print(Q)
        for _ in range(self.num_simulates):
            max_uct = 0
            for k, v in Q.items():
                if v[-1] > max_uct:
                    max_key = k
                    max_uct = v[-1]
            pos, dir = np.array([Q[max_key][2], Q[max_key][3]], dtype=int), Q[max_key][4]
            chess_board_clone = deepcopy(chess_board)
            self.fakeworld.turn = 0
            self.fakeworld.board_size = chess_board.shape[0]
            self.fakeworld.max_step = (chess_board.shape[0]+1)//2
            self.fakeworld.p1_pos = np.asarray(pos, dtype=int)
            self.fakeworld.p0_pos = np.asarray(adv_pos, dtype=int)
            self.fakeworld.chess_board = chess_board_clone
            r, c = pos
            self.fakeworld.set_barrier(r, c, dir)

            steps = 0
            #is_end, p0_score, p1_score = fakeworld.step()
            is_end, p0_score, p1_score = self.fakeworld.check_endgame()
            while not is_end and steps < self.search_steps:
                is_end, p0_score, p1_score = self.fakeworld.step()
                steps += 1
            Q[max_key][0] += p1_score/(p1_score+p0_score)
            Q[max_key][1] += 1
            Q[max_key][-1] = self.compute_uct(Q[max_key][0], Q[max_key][1], self.num_simulates)
        #print(Q)

        max_visit = 0
        for k, v in Q.items():
            if v[1] > max_uct:
                max_key = k
                max_uct = v[1]

        my_pos, my_dir = (Q[max_key][2], Q[max_key][3]), Q[max_key][4]
        return my_pos, my_dir



