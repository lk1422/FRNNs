import torch
import random
from typing import List


SOS = "<SOS>"
EOS = "<EOS>"
PAD = "<PAD>"

class Arithmetic:
    def __init__(self, max_val=1e10, test_data=False):
        #ADD DIGITS
        self.TOKENS = {str(i): i for i in range(0,10)}

        #ADD OPERATORS
        self.TOKENS["+"] = 10
        self.TOKENS["-"] = 11
        self.TOKENS["*"] = 12

        #ADD UTILITY TOKENS
        self.TOKENS["<EOS>"] = 13
        self.TOKENS["<PAD>"] = 14
        self.TOKENS["<SOS>"] = 15

        #CREATE REVERSE MAPPING
        self.VOCAB = {value:key_ for (key_,value) in self.TOKENS.items()}

        #STORE THE LARGETS POSSIBLE ARITHMETIC RESULTING VALUE
        self.maximum_operand = max_val
        self.maximum_result = max_val**2+1
        self.max_len = len(str(self.maximum_operand))*2 + 5
        self.num_tokens = len(self.TOKENS)
        self.test_data = test_data
        ##Add Tests for generalization to larger numbers
        if(test_data):
            self.maximum_operand = int(self.maximum_operand/2)

    def get_batch(self, batch_size, test=False):
        """
        Generate a random arithmetic statements for the batch
        """
        expressions = []
        results = []
        results_ = []

        for _ in range(batch_size):
            exp, res, res_ = self.generate_expression(test)
            expressions.append(exp)
            results.append(res)
            results_.append(res_)

            #assert(len(exp) == len(expressions[0]))
            #assert(len(res) == len(results[0]))

        return ( torch.Tensor(expressions).to(torch.int32) , torch.Tensor(results).to(torch.int32), torch.Tensor(results_) )

    def generate_expression(self, test=False):
        if test and self.test_data:
            """Add case to generate specific test data"""
            operand1 = random.randint(int(-self.maximum_operand) , int(self.maximum_operand) )
            operand2 = random.randint(int(-self.maximum_operand) , int(self.maximum_operand) )
        else:
            operand1 = random.randint(int(-self.maximum_operand/2) , int(self.maximum_operand/2) )
            operand2 = random.randint(int(-self.maximum_operand/2) , int(self.maximum_operand/2) )
        operation = random.choice(["+", "-", "*"])

        result = 0
        if operation == "+":
            result = operand1 + operand2
        elif operation == "-":
            result = operand1 - operand2
        elif operation == "*":
            result = operand1 * operand2

        result = str(result)
        expression = str(operand1) + operation + str(operand2)

        return ( self.tokenize_expression(expression)[1:], self.tokenize_expression(result)[:-1], self.tokenize_expression(result)[1:] )

    def tokenize_expression(self, exp:str):
        tokens = [self.TOKENS[c] for c in exp]
        tokens.append(self.TOKENS[EOS])
        tokens.extend([self.TOKENS[PAD] for _ in range(len(tokens) , self.max_len)])
        tokens = [self.TOKENS[SOS]] + tokens

        return tokens

    def get_str(self, tokens:List[int] ):
        digits = [self.VOCAB[t] for t in tokens]
        return "".join(digits)


