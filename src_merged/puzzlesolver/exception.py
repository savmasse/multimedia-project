# -*- coding: utf-8 -*-

class Unsolvable(Exception):

  def __init__ ( self, *log, data=None ):
    super(Unsolvable, self).__init__( *log )
    self.data = data

class Debug(Exception):
  def __init__ ( self, *log ):
    for l in log:
      print(l)