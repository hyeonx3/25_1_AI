import time
import threading

EMPTY = '0'
BLACK = '1'
WHITE = '2'



def timed_input(prompt, timeout):
    user_input = [None]

    def get_input():
        user_input[0] = input(prompt)

    t = threading.Thread(target=get_input)
    t.daemon = True
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None
    return user_input[0]


#현재 board에 주어진 좌표에 돌을 둔 후, 복원한다. 
def result(board, move, player):
    x,y=move
    original=board[y][x]
    board[y][x]=player
    yield board
    board[y][x]=original


#메인 함수
def Omok(player, limit_time):
    player=BLACK if player=="Black" else WHITE
    print(f"""      [GAME START]
          당신 : {"Black" if player==BLACK else "White"}, 시간 제한: {limit_time}초""")
    board=[[EMPTY for i in range(19)] for j in range(19)]
    current = BLACK

    while True:
        if current == player:
            place = timed_input("돌을 놓을 위치를 입력하세요 (ex: 3 8 ) > ", limit_time)
            if place is None:
                print("시간 초과")
                print(f"{'Black' if opponent(current) == BLACK else 'White'} 승리")
                break
            try:
                x, y = map(int, place.strip().split())
                if board[y][x] != EMPTY:
                    print("해당 위치에 이미 돌이 있습니다.")
                    continue
                move = (x, y)
            except:
                print("잘못된 입력입니다.")
                continue
            
        else:
            print("상대가 수를 두는 중입니다.")
            move = Iterative_Deepening_ABSearch(board, current, limit_time)

        board[move[1]][move[0]]=current
        show_board(board)

        if is_Terminal(board):
            print(f"{"Black" if current==BLACK else "White"} 승리")
            break

        current = opponent(current)

#해당 위치의 evaluation function 값을 return한다. 
def Evaluate_Position(board, x, y, player):
    for new_board in result(board, (x, y), player):
        pattern=count(new_board, x, y)

    score = 0
    for len,open in pattern:
        if len>= 5:
            return float('inf')
        elif len== 4:
            score+= 10000 if open== 2 else 5000
        elif len== 3:
            score += 3000 if open== 2 else 1000
        elif len== 2:
            score += 500 if open== 2 else 100
        elif len== 1:
            score += 10
    return score


#현재 전체 바둑판의 evaluation function 값을 계산한다. 나(current)에게 유리한 수는 양수, 상대에게 유리한 수는 음수로 합산한다. 
def Evaluate_State(board, player):
    score = 0
    for y in range(19):
        for x in range(19):
            if board[y][x] == player:
                score += Evaluate_Position(board, x, y, player)
            elif board[y][x] ==opponent(player):
                score -= Evaluate_Position(board, x, y,opponent(player))
    return score


def opponent(player):
    return BLACK if player == WHITE else WHITE

#move ordering을 위한 함수. 
#치명적인 수가 발견되면 즉시 return, 치명적인 수가 발견되지 않았을 시엔 Evaluate_Position의 return 값이 큰 순서대로 정렬한 뒤 상위 10개 후보만 return한다. 
def Ordering(board, player):
    candidates = generate_candidates(board, player)
    for x, y in candidates:
        if is_critical(board, x, y,player ):
            return [(x, y)]
        elif is_critical(board, x, y,opponent(player)):
            return [(x, y)]
    sorted_candidates = sorted(candidates, key=lambda pos: Evaluate_Position(board, pos[0], pos[1], player), reverse=True)
    return sorted_candidates[:10]


def is_critical(board, x, y, player):
    board[y][x] = player
    pattern = count(board, x, y)
    board[y][x] = EMPTY

    for length, open_ends in pattern:
        if length >= 5:
            return True  
        if length == 4:
            return True  
        if length == 3:
            return True  
    return False





def Iterative_Deepening_ABSearch(board, player, time_limit):
    start_time = time.time()
    best=None
    depth_limit = 1
    for x, y in generate_candidates(board, player):
        if is_critical(board, x, y, player):
            return (x, y)

    while time.time() - start_time < time_limit:
        value, move=Max_Val(board, player, float('-inf'), float('inf'), 0, depth_limit, start_time, time_limit)
        if move:
            best= move
        if value == float('inf'):
            break
        depth_limit += 1
    return best if best else (9, 9)


#오목이 완성되어 게임이 종료되었는지를 판단하는 함수
def is_Terminal(board):
    for y in range(19):
        for x in range(19):
            if board[y][x]!=EMPTY:
                pattern = count(board, x, y)
                for len,_ in pattern:
                    if len >= 5:
                        return True
    return False

def Max_Val(board, player, alpha, beta, current_depth, limit, start_time, time_limit):
    if is_Terminal(board) or current_depth == limit or time.time() - start_time > time_limit:
        return Evaluate_State(board, player), None

    v = float('-inf')
    move = None

    for x, y in Ordering(board, player):
        for new_board in result(board, (x, y), player):
            v2,a2 = Min_Val(new_board, opponent(player), alpha, beta, current_depth + 1, limit, start_time, time_limit)
        if v2 > v:
            v = v2
            move=(x, y)
            alpha=max(alpha, v)
        if v >= beta:
            return v,move
    return v,move

def Min_Val(board, player, alpha, beta, current_depth, limit, start_time, time_limit):
    if is_Terminal(board) or current_depth == limit or time.time() - start_time > time_limit:
        return Evaluate_State(board, player), None

    v = float('inf')
    move = None
    for x, y in Ordering(board, player):
        for new_board in result(board, (x, y), player):
            v2,a2= Max_Val(new_board, opponent(player), alpha, beta, current_depth + 1, limit, start_time, time_limit)
        if v2 < v:
            v = v2
            move = (x, y)
            beta = min(beta, v)
        if v <= alpha:
            return v, move
    return v, move


def generate_candidates(board, player):
    candidates = set()
    for y in range(19):
        for x in range(19):
            if board[y][x]!=EMPTY:
                for xi in range(-2, 3):
                    for yi in range(-2, 3):
                        nx, ny = x +xi, y+yi
                        if 0 <= nx < 19 and 0 <= ny < 19 and board[ny][nx] == EMPTY:
                            candidates.add((nx, ny))
    return list(candidates)

def show_board(board):
    print('   ' + ''.join(f"{i:2}" for i in range(len(board[0]))))
    for y, row in enumerate(board):
        line = f"{y:2} "
        for x in range(len(row)):
            if board[y][x] == BLACK:
                line += " ●"
            elif board[y][x] == WHITE:
                line += " ○"
            else:
                line += " ."
        print(line)

def count(board, x, y):
    current = board[y][x]
    temp = []

    def check(dx, dy):
        cnt1 = cnt2 = 0
        open_ends = 2
        i, j = x + dx, y + dy
        while 0 <= i < 19 and 0 <= j < 19 and board[j][i] == current:
            cnt1 += 1
            i += dx
            j += dy
        if not (0 <= i < 19 and 0 <= j < 19 and board[j][i] == EMPTY):
            open_ends -= 1

        i, j = x - dx, y - dy
        while 0 <= i < 19 and 0 <= j < 19 and board[j][i] == current:
            cnt2 += 1
            i -= dx
            j -= dy
        if not (0 <= i < 19 and 0 <= j < 19 and board[j][i] == EMPTY):
            open_ends -= 1

        return 1 + cnt1 + cnt2, open_ends

    for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        temp.append(check(dx, dy))

    return temp
