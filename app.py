import copy
import random

class Piece:
    def __init__(self, color, name):
        self.color = color
        self.name = name
        self.has_moved = False  # For castling and pawn initial moves

    def __repr__(self):
        return self.color[0].upper() + self.name

class Board:
    def __init__(self):
        self.grid = self.create_board()
        self.en_passant_target = None  # Square eligible for en passant capture
        self.halfmove_clock = 0  # For 50-move rule
        self.fullmove_number = 1

    def create_board(self):
        empty = None
        b, w = 'black', 'white'
        grid = [[empty]*8 for _ in range(8)]

        # Pawns
        for i in range(8):
            grid[1][i] = Piece(b, 'P')
            grid[6][i] = Piece(w, 'P')

        # Rooks
        grid[0][0] = grid[0][7] = Piece(b, 'R')
        grid[7][0] = grid[7][7] = Piece(w, 'R')

        # Knights
        grid[0][1] = grid[0][6] = Piece(b, 'N')
        grid[7][1] = grid[7][6] = Piece(w, 'N')

        # Bishops
        grid[0][2] = grid[0][5] = Piece(b, 'B')
        grid[7][2] = grid[7][5] = Piece(w, 'B')

        # Queens
        grid[0][3] = Piece(b, 'Q')
        grid[7][3] = Piece(w, 'Q')

        # Kings
        grid[0][4] = Piece(b, 'K')
        grid[7][4] = Piece(w, 'K')

        return grid

    def print_board(self):
        print("  a  b  c  d  e  f  g  h")
        for i, row in enumerate(self.grid):
            print(8 - i, end=' ')
            for cell in row:
                print(cell if cell else '--', end=' ')
            print(8 - i)
        print("  a  b  c  d  e  f  g  h\n")

    def is_in_bounds(self, r, c):
        return 0 <= r < 8 and 0 <= c < 8

    def get_piece(self, pos):
        r, c = pos
        if not self.is_in_bounds(r, c):
            return None
        return self.grid[r][c]

    def set_piece(self, pos, piece):
        r, c = pos
        self.grid[r][c] = piece

    def move_piece(self, start, end, promotion_choice=None):
        sr, sc = start
        er, ec = end
        piece = self.get_piece(start)
        target = self.get_piece(end)

        if not piece or piece.color == (target.color if target else None):
            return False, "No piece or can't capture own piece"

        legal, reason = self.is_legal_move(start, end, piece)
        if not legal:
            return False, reason

        # Handle en passant capture
        if piece.name == 'P' and (ec, er) == self.en_passant_target:
            if piece.color == 'white':
                captured_pos = (er + 1, ec)
            else:
                captured_pos = (er - 1, ec)
            self.set_piece(captured_pos, None)

        # Move piece
        self.set_piece(end, piece)
        self.set_piece(start, None)

        # Pawn promotion
        if piece.name == 'P' and (er == 0 or er == 7):
            promotion_piece = promotion_choice or 'Q'
            piece.name = promotion_piece.upper()

        # Castling move update (move rook)
        if piece.name == 'K' and abs(ec - sc) == 2:
            if ec == 6:  # Kingside
                rook_start = (sr, 7)
                rook_end = (sr, 5)
            else:  # Queenside
                rook_start = (sr, 0)
                rook_end = (sr, 3)
            rook = self.get_piece(rook_start)
            self.set_piece(rook_end, rook)
            self.set_piece(rook_start, None)
            rook.has_moved = True

        # Update flags
        piece.has_moved = True

        # Update en passant target
        self.en_passant_target = None
        if piece.name == 'P' and abs(er - sr) == 2:
            self.en_passant_target = (ec, (er + sr) // 2)

        # Update halfmove clock
        if piece.name == 'P' or target is not None:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        # Update fullmove number if black just moved
        if piece.color == 'black':
            self.fullmove_number += 1

        return True, "Move successful"

    def is_legal_move(self, start, end, piece):
        sr, sc = start
        er, ec = end
        target = self.get_piece(end)

        if target and target.color == piece.color:
            return False, "Can't capture own piece"

        dr = er - sr
        dc = ec - sc

        if piece.name == 'P':  # Pawn
            direction = -1 if piece.color == 'white' else 1
            start_row = 6 if piece.color == 'white' else 1

            # Normal forward move
            if dc == 0:
                if dr == direction and target is None:
                    return True, ""
                if sr == start_row and dr == 2 * direction and target is None and self.get_piece((sr + direction, sc)) is None:
                    return True, ""
                return False, "Illegal pawn forward move"

            # Capture diagonally
            if abs(dc) == 1 and dr == direction:
                if target and target.color != piece.color:
                    return True, ""
                # En passant capture
                if (ec, er) == self.en_passant_target:
                    return True, ""
                return False, "Illegal pawn capture"

            return False, "Illegal pawn move"

        elif piece.name == 'R':  # Rook
            if dr != 0 and dc == 0 or dr == 0 and dc != 0:
                if self.is_path_clear(start, end):
                    return True, ""
                else:
                    return False, "Path not clear for rook"
            return False, "Illegal rook move"

        elif piece.name == 'N':  # Knight
            if (abs(dr), abs(dc)) in [(2,1),(1,2)]:
                return True, ""
            return False, "Illegal knight move"

        elif piece.name == 'B':  # Bishop
            if abs(dr) == abs(dc):
                if self.is_path_clear(start, end):
                    return True, ""
                else:
                    return False, "Path not clear for bishop"
            return False, "Illegal bishop move"

        elif piece.name == 'Q':  # Queen
            if dr == 0 or dc == 0 or abs(dr) == abs(dc):
                if self.is_path_clear(start, end):
                    return True, ""
                else:
                    return False, "Path not clear for queen"
            return False, "Illegal queen move"

        elif piece.name == 'K':  # King
            if max(abs(dr), abs(dc)) == 1:
                return True, ""
            # Castling
            if dr == 0 and abs(dc) == 2:
                return self.can_castle(piece.color, start, end)
            return False, "Illegal king move"

        return False, "Unknown piece move"

    def can_castle(self, color, start, end):
        sr, sc = start
        er, ec = end

        row = 7 if color == 'white' else 0
        if sr != row or er != row:
            return False, "Castling must be on home rank"

        king = self.get_piece((row, 4))
        if king is None or king.has_moved:
            return False, "King has moved"

        if ec == 6:  # Kingside
            rook = self.get_piece((row, 7))
            if rook is None or rook.name != 'R' or rook.color != color or rook.has_moved:
                return False, "Rook has moved or missing"
            # Check path clear
            if self.get_piece((row, 5)) or self.get_piece((row,6)):
                return False, "Path blocked"
            # Check if squares under attack
            if self.is_square_attacked((row, 4), color) or self.is_square_attacked((row, 5), color) or self.is_square_attacked((row, 6), color):
                return False, "Squares under attack"
            return True, ""
        elif ec == 2:  # Queenside
            rook = self.get_piece((row, 0))
            if rook is None or rook.name != 'R' or rook.color != color or rook.has_moved:
                return False, "Rook has moved or missing"
            # Check path clear
            if self.get_piece((row, 1)) or self.get_piece((row, 2)) or self.get_piece((row, 3)):
                return False, "Path blocked"
            # Check if squares under attack
            if self.is_square_attacked((row, 4), color) or self.is_square_attacked((row, 3), color) or self.is_square_attacked((row, 2), color):
                return False, "Squares under attack"
            return True, ""

        return False, "Invalid castling move"

    def is_square_attacked(self, square, color):
        # Return True if square is attacked by any enemy piece
        enemy_color = 'black' if color == 'white' else 'white'
        for r in range(8):
            for c in range(8):
                p = self.grid[r][c]
                if p and p.color == enemy_color:
                    legal, _ = self.is_legal_move((r,c), square, p)
                    if legal:
                        return True
        return False

    def is_path_clear(self, start, end):
        sr, sc = start
        er, ec = end
        dr = er - sr
        dc = ec - sc

        step_r = (dr // abs(dr)) if dr != 0 else 0
        step_c = (dc // abs(dc)) if dc != 0 else 0

        r, c = sr + step_r, sc + step_c
        while (r, c) != (er, ec):
            if self.get_piece((r, c)) is not None:
                return False
            r += step_r
            c += step_c
        return True

    def find_king(self, color):
        for r in range(8):
            for c in range(8):
                p = self.grid[r][c]
                if p and p.name == 'K' and p.color == color:
                    return (r, c)
        return None

    def in_check(self, color):
        king_pos = self.find_king(color)
        if king_pos is None:
            return True  # No king means checkmate technically
        return self.is_square_attacked(king_pos, color)

    def has_any_legal_move(self, color):
        for r in range(8):
            for c in range(8):
                p = self.grid[r][c]
                if p and p.color == color:
                    for er in range(8):
                        for ec in range(8):
                            legal, _ = self.is_legal_move((r,c), (er,ec), p)
                            if legal:
                                # Try move and check king safety
                                temp_board = copy.deepcopy(self)
                                temp_board.set_piece((er,ec), p)
                                temp_board.set_piece((r,c), None)
                                if not temp_board.in_check(color):
                                    return True
        return False

    def is_stalemate(self, color):
        return not self.in_check(color) and not self.has_any_legal_move(color)

    def is_checkmate(self, color):
        return self.in_check(color) and not self.has_any_legal_move(color)

    def evaluate(self):
        # Simple material evaluation for AI (positive white, negative black)
        piece_values = {'P':1, 'N':3, 'B':3, 'R':5, 'Q':9, 'K':1000}
        score = 0
        for r in range(8):
            for c in range(8):
                p = self.grid[r][c]
                if p:
                    val = piece_values.get(p.name.upper(),0)
                    score += val if p.color == 'white' else -val
        return score

class Game:
    def __init__(self, vs_ai=False):
        self.board = Board()
        self.turn = 'white'
        self.vs_ai = vs_ai
        self.history = []
        self.draw_counter = 0  # 50 move rule counter

    def save_state(self):
        self.history.append(copy.deepcopy(self.board))

    def load_state(self):
        if self.history:
            self.board = self.history.pop()

    def parse_move(self, move):
        if len(move) < 4:
            return None, None
        start = notation_to_index(move[0:2])
        end = notation_to_index(move[2:4])
        promotion = move[4].upper() if len(move) == 5 else None
        return start, end, promotion

    def player_move(self):
        while True:
            self.board.print_board()
            prompt = f"{self.turn.capitalize()}'s move (e.g. e2e4"
            if self.turn == 'white':
                prompt += ", promotion e7e8Q to promote to Queen"
            prompt += "): "
            move = input(prompt).strip()

            start, end, promotion = self.parse_move(move)
            if start is None or end is None:
                print("Invalid input format. Use like e2e4 or e7e8Q for promotion.")
                continue

            piece = self.board.get_piece(start)
            if piece is None:
                print(f"No piece at {move[0:2]}.")
                continue
            if piece.color != self.turn:
                print(f"It's {self.turn}'s turn, not {piece.color}.")
                continue

            legal, reason = self.board.is_legal_move(start, end, piece)
            if not legal:
                print("Illegal move:", reason)
                continue

            # Make a test move to check for self check
            test_board = copy.deepcopy(self.board)
            success, msg = test_board.move_piece(start, end, promotion)
            if not success:
                print("Move failed:", msg)
                continue
            if test_board.in_check(self.turn):
                print("You cannot move into check!")
                continue

            # Commit move
            success, msg = self.board.move_piece(start, end, promotion)
            if success:
                # Reset draw counter if pawn move or capture
                if piece.name == 'P' or self.board.get_piece(end) is not None:
                    self.draw_counter = 0
                else:
                    self.draw_counter += 1
                return
            else:
                print("Move failed:", msg)

    def ai_move(self):
        print("AI thinking...")
        move = self.minimax_root(2, True)
        if move is None:
            print("AI resigns!")
            exit()
        start, end, promotion = move
        piece = self.board.get_piece(start)
        success, msg = self.board.move_piece(start, end, promotion)
        if success:
            print(f"AI moves: {index_to_notation(start)}{index_to_notation(end)}{promotion if promotion else ''}")
            return
        else:
            print("AI move failed:", msg)

    def minimax_root(self, depth, is_max):
        best_move = None
        best_value = -float('inf') if is_max else float('inf')

        moves = self.generate_all_moves(self.turn)

        if not moves:
            return None

        for move in moves:
            start, end, promotion = move
            temp_board = copy.deepcopy(self.board)
            success, _ = temp_board.move_piece(start, end, promotion)
            if not success:
                continue
            val = self.minimax(temp_board, depth -1, not is_max)
            if is_max and val > best_value:
                best_value = val
                best_move = move
            elif not is_max and val < best_value:
                best_value = val
                best_move = move
        return best_move

    def minimax(self, board, depth, is_max):
        if depth == 0:
            return board.evaluate()
        color = 'white' if is_max else 'black'
        moves = self.generate_all_moves(color, board)
        if not moves:
            # Checkmate or stalemate
            if board.in_check(color):
                return -10000 if is_max else 10000
            else:
                return 0
        best_val = -float('inf') if is_max else float('inf')
        for move in moves:
            start, end, promotion = move
            temp_board = copy.deepcopy(board)
            success, _ = temp_board.move_piece(start, end, promotion)
            if not success:
                continue
            val = self.minimax(temp_board, depth-1, not is_max)
            if is_max:
                best_val = max(best_val, val)
            else:
                best_val = min(best_val, val)
        return best_val

    def generate_all_moves(self, color, board=None):
        if board is None:
            board = self.board
        moves = []
        for r in range(8):
            for c in range(8):
                p = board.get_piece((r,c))
                if p and p.color == color:
                    for er in range(8):
                        for ec in range(8):
                            legal, _ = board.is_legal_move((r,c), (er,ec), p)
                            if legal:
                                # Test move for king safety
                                test_board = copy.deepcopy(board)
                                success, _ = test_board.move_piece((r,c), (er,ec))
                                if not success:
                                    continue
                                if not test_board.in_check(color):
                                    # Add promotion options for pawns
                                    if p.name == 'P' and (er == 0 or er == 7):
                                        for promo in ['Q', 'R', 'B', 'N']:
                                            moves.append(((r,c), (er,ec), promo))
                                    else:
                                        moves.append(((r,c), (er,ec), None))
        return moves

    def play(self):
        while True:
            if self.board.is_checkmate(self.turn):
                print(f"Checkmate! {self.opponent()} wins!")
                break
            if self.board.is_stalemate(self.turn):
                print("Stalemate! It's a draw.")
                break
            if self.draw_counter >= 50:
                print("50-move rule reached. It's a draw.")
                break

            if self.turn == 'white' or not self.vs_ai:
                self.player_move()
            else:
                self.ai_move()

            # Swap turn
            self.turn = self.opponent()

    def opponent(self):
        return 'black' if self.turn == 'white' else 'white'

def notation_to_index(notation):
    files = 'abcdefgh'
    ranks = '87654321'
    if len(notation) != 2:
        return None
    file = notation[0]
    rank = notation[1]
    if file not in files or rank not in ranks:
        return None
    return (ranks.index(rank), files.index(file))

def index_to_notation(pos):
    files = 'abcdefgh'
    ranks = '87654321'
    r, c = pos
    return files[c] + ranks[r]

if __name__ == "__main__":
    mode = input("Play against AI? (y/n): ").strip().lower()
    game = Game(vs_ai=(mode == 'y'))
    game.play()
