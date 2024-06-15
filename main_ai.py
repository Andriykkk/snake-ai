# importing libraries
import os

import pygame
import time
import random
from config import window_x, window_y, black, white, green, red, snake_speed
from agent import Agent
import matplotlib.pyplot as plt
from model import SnakeNet

# Initialising pygame
pygame.init()

# Initialise game window
pygame.display.set_caption('GeeksforGeeks Snakes')
game_window = pygame.display.set_mode((window_x, window_y))

# FPS (frames per second) controller
fps = pygame.time.Clock()

record = 0
moves = 0

# defining snake default position
snake_position = [100, 50]

# defining first 4 blocks of snake body
snake_body = [[100, 50],
              [90, 50],
              [80, 50],
              [70, 50]
              ]
# fruit position
fruit_position = [random.randrange(1, (window_x // 10)) * 10,
                  random.randrange(1, (window_y // 10)) * 10]

fruit_spawn = True

# setting default snake direction towards
# right
direction = 'RIGHT'
change_to = direction
n_games = 1

# initial score
score = 0

# Initialize variables for plotting
plot_scores = []
plot_mean_scores = []
total_score = 0
record = 0
games = []


def set_variables():
    global snake_position, snake_body, fruit_position, fruit_spawn, change_to, score, moves

    snake_position = [100, 50]
    # defining first 4 blocks of snake body
    snake_body = [[100, 50],
                  [90, 50],
                  [80, 50],
                  [70, 50]
                  ]
    # fruit position
    fruit_position = [random.randrange(1, (window_x // 10)) * 10,
                      random.randrange(1, (window_y // 10)) * 10]

    fruit_spawn = True
    moves = 0

    # setting default snake direction towards
    # right
    direction = 'DOWN'
    change_to = direction

    # initial score
    score = 0


# displaying Score function
def show_score(choice, color, font, size):
    # creating font object score_font
    score_font = pygame.font.SysFont(font, size)

    # create the display surface object
    # score_surface
    score_surface = score_font.render('Score : ' + str(score), True, color)

    # create a rectangular object for the text
    # surface object
    score_rect = score_surface.get_rect()

    # displaying text
    game_window.blit(score_surface, score_rect)


# game over function
def game_over():
    global score, n_games, agent, mode, record
    # creating font object my_font
    my_font = pygame.font.SysFont('times new roman', 50)

    # creating a text surface on which text
    # will be drawn
    game_over_surface = my_font.render(
        'Your Score is : ' + str(score), True, red)

    update_plot(score)
    # create a rectangular object for the text
    # surface object
    game_over_rect = game_over_surface.get_rect()

    # setting position of the text
    game_over_rect.midtop = (window_x / 2, window_y / 4)

    # blit will draw the text on screen
    game_window.blit(game_over_surface, game_over_rect)
    pygame.display.flip()
    n_games += 1
    set_variables()
    if mode != 'watch':
        agent.train_long_memory()

        if score > record:
            record = score
            agent.model.save()
            print("model saved with record: " + record)

    return True


def change_direction():
    global direction, change_to

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                change_to = 'UP'
            if event.key == pygame.K_DOWN:
                change_to = 'DOWN'
            if event.key == pygame.K_LEFT:
                change_to = 'LEFT'
            if event.key == pygame.K_RIGHT:
                change_to = 'RIGHT'

        # If two keys pressed simultaneously
        # we don't want snake to move into two
        # directions simultaneously
    if change_to == 'UP' and direction != 'DOWN':
        direction = 'UP'
    if change_to == 'DOWN' and direction != 'UP':
        direction = 'DOWN'
    if change_to == 'LEFT' and direction != 'RIGHT':
        direction = 'LEFT'
    if change_to == 'RIGHT' and direction != 'LEFT':
        direction = 'RIGHT'

        # Moving the snake
    if direction == 'UP':
        snake_position[1] -= 10
    if direction == 'DOWN':
        snake_position[1] += 10
    if direction == 'LEFT':
        snake_position[0] -= 10
    if direction == 'RIGHT':
        snake_position[0] += 10


def change_direction_ai(action):
    global change_to, direction

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                change_to = 'UP'
            if event.key == pygame.K_DOWN:
                change_to = 'DOWN'
            if event.key == pygame.K_LEFT:
                change_to = 'LEFT'
            if event.key == pygame.K_RIGHT:
                change_to = 'RIGHT'

    # action = [left, straight, right]
    if action == [1, 0, 0]:
        if direction == 'UP':
            change_to = 'LEFT'
        elif direction == 'DOWN':
            change_to = 'RIGHT'
        elif direction == 'LEFT':
            change_to = 'DOWN'
        elif direction == 'RIGHT':
            change_to = 'UP'
    elif action == [0, 1, 0]:
        change_to = direction
    elif action == [0, 0, 1]:
        if direction == 'UP':
            change_to = 'RIGHT'
        elif direction == 'DOWN':
            change_to = 'LEFT'
        elif direction == 'LEFT':
            change_to = 'UP'
        elif direction == 'RIGHT':
            change_to = 'DOWN'

    direction = change_to

    if direction == 'UP':
        snake_position[1] -= 10
    if direction == 'DOWN':
        snake_position[1] += 10
    if direction == 'LEFT':
        snake_position[0] -= 10
    if direction == 'RIGHT':
        snake_position[0] += 10


def create_plot_window():
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel('Games')
    ax.set_ylabel('Score')
    ax.set_title('Score Progression')
    return fig, ax


def update_plot(score):
    global total_score, record, n_games, fig, ax
    plot_scores.append(score)
    total_score += score
    mean_score = total_score / n_games
    plot_mean_scores.append(mean_score)
    games.append(n_games)

    ax.clear()
    ax.plot(games, plot_scores, label='Score', color='r')
    ax.plot(games, plot_mean_scores, label='Mean Score', color='g')
    ax.set_xlabel('Games')
    ax.set_ylabel('Score')
    ax.legend()
    ax.set_title('Score Progression')
    fig.canvas.draw()
    fig.canvas.flush_events()


def play_step(action):
    global change_to, direction, fruit_position, fruit_spawn, score, moves
    change_direction_ai(action)
    reward = 0

    # Snake body growing mechanism
    # if fruits and snakes collide then scores
    # will be incremented by 10
    snake_body.insert(0, list(snake_position))
    if snake_position[0] == fruit_position[0] and snake_position[1] == fruit_position[1]:
        reward = 10
        score += 10
        fruit_spawn = False
    else:
        snake_body.pop()

    if not fruit_spawn:
        fruit_position = [random.randrange(1, (window_x // 10)) * 10,
                          random.randrange(1, (window_y // 10)) * 10]

    fruit_spawn = True
    game_window.fill(black)
    is_game_over = False

    for pos in snake_body:
        pygame.draw.rect(game_window, green,
                         pygame.Rect(pos[0], pos[1], 10, 10))
    pygame.draw.rect(game_window, white, pygame.Rect(
        fruit_position[0], fruit_position[1], 10, 10))

    if moves > score * 10 + 200:
        is_game_over = True
        game_over()

    # Game Over conditions
    if snake_position[0] < 0 or snake_position[0] > window_x - 10:
        game_over()
        is_game_over = True
    if snake_position[1] < 0 or snake_position[1] > window_y - 10:
        game_over()
        is_game_over = True

    # Touching the snake body
    for block in snake_body[1:]:
        if snake_position[0] == block[0] and snake_position[1] == block[1]:
            game_over()
            is_game_over = True

    if is_game_over:
        reward = -10

    return reward, is_game_over


agent = Agent(SnakeNet(15, 3), 0.001, 0.8, 0.1)


def ask_user_mode():
    print("Choose mode:")
    print("1. Start model learning")
    print("2. Continue learning")
    print("3. Watch how the snake plays")
    choice = input("Enter 1, 2, or 3: ")

    if choice == '1':
        return 'start_learning'
    elif choice == '2':
        return 'continue_learning'
    elif choice == '3':
        return 'watch'
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
        return ask_user_mode()


if __name__ == "__main__":
    mode = ask_user_mode()
    fig, ax = create_plot_window()

    if mode == 'continue_learning' or mode == 'watch':
        if os.path.exists('model.pth'):
           agent.load_model('model.pth')
        else:
            print("No saved model found. Starting fresh.")
            mode = 'start_learning'

    if mode == 'watch':
        agent.epsilon = 0.0

    # Main Function
    while True:
        # handling key events
        # change_direction()
        if mode != 'watch':
            state_old = agent.get_state(snake_position, fruit_position, direction, snake_body, score)
        predict_action = agent.get_action(state_old, n_games)

        moves += 1

        reward, done = play_step(predict_action)

        state_new = agent.get_state(snake_position, fruit_position, direction, snake_body, score)

        if mode != 'watch':
            agent.train_short_memory(state_old, predict_action, reward, state_new, done)
            agent.remember(state_old, predict_action, reward, state_new, done)

        # displaying score continuously
        show_score(1, white, 'times new roman', 20)

        # Refresh game screen
        pygame.display.update()

        # Frame Per Second /Refresh Rate
        fps.tick(snake_speed)
