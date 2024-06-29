# Imports
import pygame
import sys
import random
import math


class FlappyBird:

    def __init__(self):

        self.pipe_is_spawned = False

        self.screen = pygame.display.set_mode((1152, 1024))
        self.clock = pygame.time.Clock()
        self.game_font = pygame.font.SysFont('comicsans', 60)

        # Setting Game Variables
        self.gravity = 0.6
        self.bird_movement = 0
        self.game_active = False
        self.score = 0
        self.previous_score = 0
        self.high_score = 0
        self.speed = 4

        # Setting variables for game asset images
        self.bg_surface = pygame.image.load('imgs/bg.png').convert()
        original_width, original_height = self.bg_surface.get_size()

        new_width = 1152
        scale_factor = new_width / original_width
        new_height = int(original_height * scale_factor)
        self.bg_surface = pygame.transform.scale(self.bg_surface, (new_width, new_height))

        self.floor_surface = pygame.image.load('imgs/base.png').convert()
        original_width, original_height = self.floor_surface.get_size()

        # Scale to new width (1152) while maintaining aspect ratio
        new_width = 1152
        scale_factor = new_width / original_width
        new_height = int(original_height * scale_factor)

        self.floor_surface = pygame.transform.scale(self.floor_surface, (new_width, new_height))

        self.floor_x_pos = 0

        self.bird_downflap = pygame.transform.scale2x(
            pygame.image.load('imgs/bird1.png')).convert_alpha()
        self.bird_midflap = pygame.transform.scale2x(
            pygame.image.load('imgs/bird2.png')).convert_alpha()
        self.bird_upflap = pygame.transform.scale2x(
            pygame.image.load('imgs/bird3.png')).convert_alpha()
        self.bird_frames = [self.bird_downflap, self.bird_midflap, self.bird_upflap]
        self.bird_index = 0
        self.bird_surface = self.bird_frames[self.bird_index]
        self.bird_rect = self.bird_surface.get_rect(center=(100, 512))

        self.BIRDFLAP = pygame.USEREVENT + 1
        pygame.time.set_timer(self.BIRDFLAP, 200)

        self.SPEEDUP = pygame.USEREVENT + 2
        pygame.time.set_timer(self.SPEEDUP, 40000)

        self.pipe_surface = pygame.image.load('imgs/pipe.png').convert()
        self.pipe_surface = pygame.transform.scale2x(self.pipe_surface)
        self.pipe_list = []
        self.SPAWNPIPE = pygame.USEREVENT
        pygame.time.set_timer(self.SPAWNPIPE, 1000)
        self.pipe_height = [400, 525, 600, 750, 800, 450, 500, 575, 650, 700]  # Add or adjust heights as needed

        self.point_collider = pygame.Surface((10, 300))
        self.point_collider.set_alpha(0)
        self.collider_list = []
        self.rewardMultiplier = 1
        self.INCREASEREWARD = pygame.USEREVENT
        pygame.time.set_timer(self.INCREASEREWARD, 1000)

    @staticmethod
    def draw_floor(self):
        self.screen.blit(self.floor_surface, (self.floor_x_pos, 900))
        self.screen.blit(self.floor_surface, (self.floor_x_pos + 1152, 900))

    @staticmethod
    def create_pipe(self):
        # creates pipes
        random_pipe_pos = random.choice(self.pipe_height)
        bottomPipe = self.pipe_surface.get_rect(midtop=(1152, random_pipe_pos))
        topPipe = self.pipe_surface.get_rect(midbottom=(1152, random_pipe_pos - 300))
        point_collider_rect = self.point_collider.get_rect(midright=(1152, random_pipe_pos - 150))  # Adjusted position for collision
        return bottomPipe, topPipe, point_collider_rect


    @staticmethod
    def move_pipes(self, pipes, colliders):
        for pipe in pipes:
            pipe.centerx -= self.speed
        for collider in colliders:
            collider.centerx -= self.speed
        return pipes, colliders

    @staticmethod
    def draw_pipes(self, pipes, colliders):
        for pipe in pipes:
            if pipe.bottom >= 1024:
                self.screen.blit(self.pipe_surface, pipe)
            else:
                flip_pipe = pygame.transform.flip(self.pipe_surface, False, True)
                self.screen.blit(flip_pipe, pipe)

        for collider in colliders:
            self.screen.blit(self.point_collider, collider)

    @staticmethod
    def check_collision(self, pipes):
        for pipe in pipes:
            if self.bird_rect.colliderect(pipe):
                return False

        if self.bird_rect.top <= -100 or self.bird_rect.bottom >= 900:
            return False

        return True

    @staticmethod
    def check_pipe_reached(self, colliders):
        pipe_passed = False
        for collider in colliders:
            if self.bird_rect.centerx > collider.centerx and not pipe_passed:
                colliders.remove(collider)
                self.pipe_is_spawned = False
                return colliders, True  # Return updated list and flag indicating point scored
        return colliders, False  # Return original list and flag indicating no point scored


    @staticmethod
    def rotate_bird(self, bird):
        new_bird = pygame.transform.rotozoom(bird, -self.bird_movement * 2, 1)
        return new_bird

    @staticmethod
    def bird_animation(self):
        new_bird = self.bird_frames[self.bird_index]
        new_bird_rect = new_bird.get_rect(center=(100, self.bird_rect.centery))
        return new_bird, new_bird_rect

    @staticmethod
    def score_display(self):
        score_surface = self.game_font.render(str(self.score), True, (255, 255, 255))
        score_rect = score_surface.get_rect(center=(self.screen.get_width() // 2, 100))
        self.screen.blit(score_surface, score_rect)

    @staticmethod
    def is_done(self):
        if not self.game_active:
            self.pipe_is_spawned = False
        return not self.game_active

    @staticmethod
    def get_score(self):
        return self.score

    @staticmethod
    def start_game(self):
        self.game_active = True
        self.pipe_list.clear()
        self.collider_list.clear()
        self.bird_rect.center = (100, 512)
        self.bird_movement = 0
        self.score = 0
        self.speed = 15
        self.pipe_is_spawned = False

    @staticmethod
    def check_event(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if action == 1:
                self.bird_movement = 0
                self.bird_movement -= 15

            if event.type == self.SPAWNPIPE and self.game_active:
                bottom_pipe, top_pipe, win_collider = self.create_pipe(self)
                self.pipe_list.append(bottom_pipe)
                self.pipe_list.append(top_pipe)
                self.collider_list.append(win_collider)
                self.pipe_is_spawned = True

            if event.type == self.BIRDFLAP:
                if self.bird_index < 2:
                    self.bird_index += 1
                else:
                    self.bird_index = 0

                self.bird_surface, self.bird_rect = self.bird_animation(self)

            if event.type == self.SPEEDUP and self.game_active:
                if self.speed < 8:
                    self.speed += 0.05

            if event.type == self.INCREASEREWARD and self.game_active:
                self.rewardMultiplier += 0.01

    @staticmethod
    def is_game_active(self):
        # Bird
        self.bird_movement += self.gravity
        rotated_bird = self.rotate_bird(self, self.bird_surface)
        self.bird_rect.centery += self.bird_movement
        self.screen.blit(rotated_bird, self.bird_rect)
        self.game_active = self.check_collision(self, self.pipe_list)
        collider_list, pipe_reached = self.check_pipe_reached(self, self.collider_list)
        if pipe_reached:
            self.score += 1

        # Pipes
        pipe_list, collider_list = self.move_pipes(self, self.pipe_list, self.collider_list)
        self.draw_pipes(self, pipe_list, collider_list)

        self.score_display(self)

    @staticmethod
    def draw_frame(self):
        self.screen.blit(self.bg_surface, (0, 0))
        self.is_game_active(self)
        
        # Floor
        self.floor_x_pos -= 4
        self.draw_floor(self)
        if self.floor_x_pos <= -576:
            self.floor_x_pos = 0

        pygame.display.update()
        self.clock.tick(60)

    @staticmethod
    def get_reward(self, action):
        point = 0.1 * self.rewardMultiplier
        # Reward structure that should keep bird from flapping stright up to the ceiling and push agent to go for highest score
        if action == 1:
            point -= 0.01 * self.rewardMultiplier # Small penalty for flapping to counteract unecessary actions 
        elif not self.check_collision:
            point -= 100 * self.rewardMultiplier
        elif self.check_pipe_reached:
            point += 50 * self.rewardMultiplier
        elif self.bird_rect.top <= 0:
            point -= 1
        elif self.score > self.previous_score:
            point += 100

        return point

    @staticmethod
    def get_state(self):
        if self.pipe_is_spawned:
            bird_y = self.bird_rect.centery / 1024  # Normalize bird y-position
            collider_x = self.collider_list[0].centerx / 1152  # Normalize collider x-position
            collider_y = self.collider_list[0].centery / 1024  # Normalize collider y-position
            distance_to_collider = math.sqrt((self.collider_list[0].centerx - self.bird_rect.centerx) ** 2 + (self.collider_list[0].centery - self.bird_rect.centery) ** 2) / 1024  # Normalize distance
            vertical_distance_to_collider = (self.collider_list[0].centery - self.bird_rect.centery) / 1024  # Normalize vertical distance
            horizontal_distance_to_collider = (self.collider_list[0].centerx - self.bird_rect.centerx) / 1152  # Normalize horizontal distance
            bird_movement = self.bird_movement / 20.0  # Normalize bird movement
            distance_to_ground = (900 - self.bird_rect.bottom) / 900.0  # Normalize distance to ground
            distance_to_collider_vertical = (900 - self.collider_list[0].centery) / 900.0  # Normalize distance to collider vertical

            position = [bird_y, collider_x, collider_y, distance_to_collider, vertical_distance_to_collider, horizontal_distance_to_collider, bird_movement, distance_to_ground, distance_to_collider_vertical]
        else:
            position = [self.bird_rect.centery / 1024, 0, 0, 0, 0, 0, self.bird_movement / 20.0, (900 - self.bird_rect.bottom) / 900.0, 50 / 900.0]  # Normalize all other states

        return position


    @staticmethod
    def step(self, action):
        self.check_event(self, action)
        self.draw_frame(self)

        point = self.get_reward(self, action)

        self.is_done(self)

        position = self.get_state(self)

        self.previous_score = self.score

        return point, position