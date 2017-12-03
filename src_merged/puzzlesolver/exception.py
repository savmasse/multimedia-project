# -*- coding: utf-8 -*-

class Unsolvable(Exception):
  pass

class Debug(Exception):
  def __init__ ( *log ):
    for l in log[1:]:
      print(l)