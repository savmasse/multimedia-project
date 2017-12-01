# -*- coding: utf-8 -*-

import puzzlesolver as ps

root = '../toon_sandbox/img'
allowed_types = ['2x2', 'scrambled', 'tiles']

def solve_puzzle ( path ):
  puzzle = ps.Puzzle(path)
  puzzle.extract_pieces()
  print(puzzle.correct(path))

ps.Picker(solve_puzzle, root, types=allowed_types)