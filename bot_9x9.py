"""Bot Arena 9x9: submit this entire file; no third-party packages required.

Entry point: make_move(state) -> {"from": "B2", "to": "C3"}.
Uses deterministic, bounded alpha-beta beam search and a legal root move list.
"""

import time

_NAMES = [chr(65 + c) + str(r + 1) for r in range(9) for c in range(9)]
_INDEX = {name: i for i, name in enumerate(_NAMES)}
_TYPES = {"ROCK": 1, "PAPER": 2, "SCISSORS": 3}
_BEATS = (0, 3, 1, 2)
_SWAP = (0, 2, 3, 1)
_NEIGHBORS = [
    tuple((r + dr) * 9 + c + dc
          for dc in (-1, 0, 1) for dr in (-1, 0, 1)
          if (dc or dr) and 0 <= c + dc < 9 and 0 <= r + dr < 9)
    for r in range(9) for c in range(9)
]
_DIST = [[max(abs(a % 9 - b % 9), abs(a // 9 - b // 9))
          for b in range(81)] for a in range(81)]
_WIN = 100000
_MEMORY = {}


class _BudgetReached(Exception):
    pass


class _Search:
    def __init__(self, state, memory):
        self.me = 1 if state["my_side"] == "blue" else -1
        self.start_ply = state.get("ply", 0)
        self.max_plies = state.get("max_plies", 200)
        goals = state.get("goals", {"blue": "I9", "pink": "A1"})
        self.goals = {1: _INDEX[goals["blue"]], -1: _INDEX[goals["pink"]]}
        special = state.get("special_cells", [])
        self.shields = {_INDEX[x["square"]] for x in special if x["kind"] == "shield"}
        self.swaps = {_INDEX[x["square"]] for x in special if x["kind"] == "swap"}
        self.variant = bool(special or state.get("fixed_obstacles")
                            or state.get("moving_obstacles"))
        self.fixed = {_INDEX[x] for x in state.get("fixed_obstacles", [])}
        self.moving = [(tuple(_INDEX[x] for x in item["path"]), item["every"])
                       for item in state.get("moving_obstacles", [])]
        # Older payloads may only include the current obstacle positions.
        if not self.moving:
            self.fixed.update(_INDEX[x] for x in state.get("obstacles", []))
        self.block_cache = {}
        self.repetitions = memory
        self.nodes = 0
        self.cpu_start = time.process_time()
        self.wall_start = time.perf_counter()

    def blocked(self, ply):
        if ply not in self.block_cache:
            self.block_cache[ply] = self.fixed | {
                path[(ply // every) % len(path)] for path, every in self.moving
            }
        return self.block_cache[ply]

    def signature(self, board, side, ply):
        moving = tuple(path[(ply // every) % len(path)] for path, every in self.moving)
        return (tuple(board), side, moving)

    def legal(self, board, side, ply):
        blocked = self.blocked(ply)
        moves = []
        for src, piece in enumerate(board):
            if piece * side <= 0:
                continue
            prey = _BEATS[abs(piece)]
            for dst in _NEIGHBORS[src]:
                if dst in blocked:
                    continue
                target = board[dst]
                if not target or (target * side < 0 and abs(target) == prey
                                  and dst not in self.shields):
                    moves.append((src, dst))
        return moves

    def apply(self, board, move):
        src, dst = move
        result = board[:]
        piece = result[src]
        result[src] = 0
        if dst in self.swaps:
            piece = (1 if piece > 0 else -1) * _SWAP[abs(piece)]
        result[dst] = piece
        return result

    @staticmethod
    def counts(board):
        counts = {1: [0, 0, 0, 0], -1: [0, 0, 0, 0]}
        for piece in board:
            if piece:
                counts[1 if piece > 0 else -1][abs(piece)] += 1
        return counts

    def terminal(self, board, mover, ply, counts, next_moves, repetition):
        # The priority matches the actual referee, including swap self-elimination.
        goal_piece = board[self.goals[mover]]
        winner = None
        if goal_piece * mover > 0:
            winner = mover
        elif 0 in counts[-mover][1:]:
            winner = mover
        elif self.variant and 0 in counts[mover][1:]:
            winner = -mover
        elif not next_moves:
            winner = mover
        elif repetition >= 3 or ply >= self.max_plies:
            diff = sum(counts[self.me]) - sum(counts[-self.me])
            return 0 if diff == 0 else (1 if diff > 0 else -1) * (_WIN - (ply - self.start_ply))
        if winner is None:
            return None
        return (1 if winner == self.me else -1) * (_WIN - (ply - self.start_ply))

    def evaluate(self, board, counts, ply):
        blocked = self.blocked(ply)
        scores = {1: 0, -1: 0}
        distances = {1: [], -1: []}
        for side in (1, -1):
            scores[side] = 120 * sum(counts[side])
            # Losing the final piece of ANY type loses the entire game.
            scores[side] -= sum(65 if n == 1 else 15 if n == 2 else 0
                                for n in counts[side][1:])
        for sq, piece in enumerate(board):
            if not piece:
                continue
            side = 1 if piece > 0 else -1
            kind = abs(piece)
            distances[side].append(_DIST[sq][self.goals[side]])
            if sq in self.shields:
                scores[side] += 22
            elif sq not in blocked:
                threatened = any(board[n] * side < 0
                                 and _BEATS[abs(board[n])] == kind
                                 for n in _NEIGHBORS[sq])
                if threatened:
                    scores[side] -= 55 + (100 if counts[side][kind] == 1 else 0)
            # Encourage central access without abandoning the goal race.
            scores[side] += 4 - max(abs(sq % 9 - 4), abs(sq // 9 - 4))
        for side in (1, -1):
            ds = sorted(distances[side])
            if ds:
                scores[side] -= 17 * ds[0] + 3 * sum(ds[:3])
                if ds[0] == 1:
                    scores[side] += 35
        return scores[self.me] - scores[-self.me]

    def ordered(self, board, moves, side, counts, ply):
        goal = self.goals[side]
        next_blocked = self.blocked(ply + 1)

        def priority(move):
            src, dst = move
            if dst == goal:
                return 1000000
            kind = abs(board[src])
            score = 9 * (_DIST[src][goal] - _DIST[dst][goal])
            target = board[dst]
            if target:
                score += 200 if counts[-side][abs(target)] > 1 else 100000
            if dst in self.swaps:
                if counts[side][kind] == 1:
                    score -= 200000
                kind = _SWAP[kind]
            if dst in self.shields:
                score += 25
            elif dst not in next_blocked:
                if any(n != src and board[n] * side < 0
                       and _BEATS[abs(board[n])] == kind for n in _NEIGHBORS[dst]):
                    score -= 110 + (150 if counts[side][kind] == 1 else 0)
            return score

        # Coordinates break ties deterministically, independent of input dict order.
        return sorted(moves, key=lambda m: (-priority(m), m))

    def visit(self, board, mover, ply, depth, alpha, beta):
        self.nodes += 1
        # A fixed node ceiling is the normal stop; clocks are emergency guards.
        if self.nodes > 520 or (self.nodes % 16 == 0 and (
                time.process_time() - self.cpu_start > 0.065
                or time.perf_counter() - self.wall_start > 0.32)):
            raise _BudgetReached
        side = -mover
        key = self.signature(board, side, ply)
        self.repetitions[key] = self.repetitions.get(key, 0) + 1
        try:
            counts = self.counts(board)
            moves = self.legal(board, side, ply)
            end = self.terminal(board, mover, ply, counts, moves, self.repetitions[key])
            if end is not None:
                return end
            if depth == 0:
                return self.evaluate(board, counts, ply)
            ranked = self.ordered(board, moves, side, counts, ply)
            # Beam search bounds branching; immediate wins are ordered first.
            ranked = ranked[:10 if depth >= 2 else 8]
            best = -2 * _WIN if side == self.me else 2 * _WIN
            for move in ranked:
                value = self.visit(self.apply(board, move), side, ply + 1,
                                   depth - 1, alpha, beta)
                if side == self.me:
                    best = max(best, value)
                    alpha = max(alpha, best)
                else:
                    best = min(best, value)
                    beta = min(beta, best)
                if alpha >= beta:
                    break
            return best
        finally:
            self.repetitions[key] -= 1
            if not self.repetitions[key]:
                del self.repetitions[key]

    def choose(self, board, moves):
        counts = self.counts(board)
        ranked = self.ordered(board, moves, self.me, counts, self.start_ply)
        best = ranked[0]
        scored = []
        # First score ALL root moves, so a direct win is never excluded by the beam.
        for move in ranked:
            try:
                value = self.visit(self.apply(board, move), self.me,
                                   self.start_ply + 1, 0, -2 * _WIN, 2 * _WIN)
            except _BudgetReached:
                break
            scored.append((value, move))
            if value >= _WIN - 1:
                return move
        if scored:
            scored.sort(key=lambda item: (-item[0], item[1]))
            best = scored[0][1]
        candidates = [move for _, move in scored[:16]]
        # Keep the last fully completed depth if the next iteration runs out of budget.
        for depth in (2, 3):
            iteration = []
            alpha = -2 * _WIN
            try:
                for move in candidates:
                    value = self.visit(self.apply(board, move), self.me,
                                       self.start_ply + 1, depth - 1, alpha, 2 * _WIN)
                    iteration.append((value, move))
                    alpha = max(alpha, value)
            except _BudgetReached:
                break
            if iteration:
                # Stable sorting preserves the stronger previous-depth ordering on ties.
                iteration.sort(key=lambda item: -item[0])
                best = iteration[0][1]
                candidates = [move for _, move in iteration]
        return best


def make_move(state):
    """Select a legal move for either side in standard and variant 9x9 games."""
    legal = state.get("legal_moves", [])
    if not legal:
        # The referee normally ends the game before calling a bot with no moves.
        return {"from": "A1", "to": "A1"}
    board = [0] * 81
    for sq, piece in state["board"].items():
        board[_INDEX[sq]] = _TYPES[piece["type"]] * (1 if piece["side"] == "blue" else -1)
    ply = state.get("ply", 0)
    identity = (state.get("seed", 0), state.get("ruleset_id"), state["my_side"])
    previous = _MEMORY.get(identity)
    if previous is None or ply <= previous["ply"] or ply != previous["ply"] + 2:
        previous = {"ply": -2, "reps": {}}
        _MEMORY.clear()
        _MEMORY[identity] = previous
    search = _Search(state, previous["reps"].copy())
    root_key = search.signature(board, search.me, ply)
    search.repetitions[root_key] = search.repetitions.get(root_key, 0) + 1
    candidates = [(_INDEX[m["from"]], _INDEX[m["to"]]) for m in legal]
    chosen = search.choose(board, candidates)
    previous["ply"] = ply
    previous["reps"] = search.repetitions.copy()
    next_board = search.apply(board, chosen)
    next_key = search.signature(next_board, -search.me, ply + 1)
    previous["reps"][next_key] = previous["reps"].get(next_key, 0) + 1
    return {"from": _NAMES[chosen[0]], "to": _NAMES[chosen[1]]}
