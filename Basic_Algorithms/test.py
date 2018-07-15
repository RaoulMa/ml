#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""


class LogicGate:
    def __init__(self, label):
        self.label = label
        self.output = None
    
    def get_label(self):
        return self.label
    
    def get_output(self):
        self.output = self.perform_gate_logic()
        return self.output
        

class AndGate(LogicGate):
    def __init__(self, label):
        LogicGate.__init__(self,label)
        self.pinA = None
        self.pinB = None
    
    def set_pins(self,pinA,pinB):
        self.pinA = pinA
        self.pinB = pinB
    
    def perform_gate_logic(self):
        return (self.pinA & self.pinB)
    

g1 = AndGate("And Gate Nr 1")
g1.set_pins(1,0)

print(g1.get_label(), g1.get_output())


g2 = AndGate("And Gate Nr 2")
print(g2.get_label())
    
        
