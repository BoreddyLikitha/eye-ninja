"""
EYE NINJA - Professional Eye Tracking Game

A Fruit Ninja-style game with improved eye tracking.
Features:
- Exponential smoothing filter
- Blink detection with debounce
- Sensitivity adjustment
- Dead zone for jitter reduction
- Debug camera view

Requirements:
    pip install pygame opencv-python numpy
"""

import cv2
import numpy as np
import pygame
import random
import math
from collections import deque
from typing import Optional, Tuple, List, Deque


# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("EYE NINJA - Professional Eye Tracking")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)

# Game States
MENU = 0
PLAYING = 1
GAME_OVER = 2
PAUSED = 3
LEVEL_COMPLETE = 4


class EyeTracker:
    """
    Improved eye tracker using OpenCV Haar cascades.
    
    Features:
    - Exponential smoothing filter
    - Blink detection with debounce
    - Sensitivity adjustment
    - Dead zone for jitter reduction
    """
    
    def __init__(
        self, 
        debug: bool = False,
        sensitivity: float = 1.0
    ) -> None:
        """Initialize eye tracker."""
        self.debug = debug
        self.sensitivity = max(0.1, min(2.0, sensitivity))
        
        # Webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Load Haar cascades
        cascade_path = cv2.data.haarcascades
        
        # Try multiple face cascades for better detection
        self.face_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_eye.xml'
        )
        
        # Gaze state
        self.gaze_x: int = WIDTH // 2
        self.gaze_y: int = HEIGHT // 2
        
        # Exponential smoothing
        self.alpha: float = 0.3 / self.sensitivity
        
        # Dead zone
        self.dead_zone: float = 0.03
        
        # Blink detection
        self.blink_detected: bool = False
        self.blink_cooldown: int = 0
        self.blink_debounce_frames: int = 8  # Require more frames for blink detection
        self.eye_open_history: Deque[bool] = deque(maxlen=5)
        
        # Tracking state
        self._face_detected: bool = False
        self._last_face_x: int = 0
        self._last_face_y: int = 0
        
    def _exponential_smooth(self, new_value: float, old_value: float) -> float:
        """Apply exponential smoothing filter."""
        return self.alpha * new_value + (1 - self.alpha) * old_value
    
    def _apply_dead_zone(self, value: float, center: float) -> float:
        """Apply dead zone around center."""
        dead_zone_pixels = WIDTH * self.dead_zone
        if abs(value - center) < dead_zone_pixels:
            return center
        return value
    
    def _get_pupil_position(
        self, 
        eye_roi: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """Detect pupil position in eye region."""
        if eye_roi.size == 0:
            return None
            
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Apply blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try multiple thresholds
        for threshold_val in [20, 30, 40, 50, 60]:
            _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY_INV)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 10 and area < (h * w * 0.4):
                        M = cv2.moments(cnt)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            if 0.2 * w < cx < 0.8 * w and 0.2 * h < cy < 0.8 * h:
                                return (cx, cy)
        
        # Fallback: center of eye
        return (w // 2, h // 2)
    
    def _detect_blink(self, eyes_detected: bool) -> bool:
        """Detect blink using eye detection."""
        self.eye_open_history.append(eyes_detected)
        
        if len(self.eye_open_history) >= self.blink_debounce_frames:
            recent = list(self.eye_open_history)[-self.blink_debounce_frames:]
            if not any(recent):  # All frames had no eyes detected
                return True
        
        return False
    
    def update(self) -> bool:
        """Update gaze tracking."""
        ret, frame = self.cap.read()
        if not ret:
            return False
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with more sensitive parameters
        faces = self.face_cascade.detectMultiScale(
            gray, 1.05, 3, minSize=(100, 100)
        )
        
        self.blink_detected = False
        self._face_detected = len(faces) > 0
        
        if len(faces) > 0:
            # Use largest face
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            x, y, w, h = faces[0]
            
            self._last_face_x, self._last_face_y = x, y
            
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 2)
            
            # Check for blink
            if self._detect_blink(len(eyes) > 0) and self.blink_cooldown == 0:
                self.blink_detected = True
                self.blink_cooldown = 30  # Longer cooldown to prevent false triggers
                # Clear history after detecting blink to prevent continuous detection
                self.eye_open_history.clear()
            
            if self.blink_cooldown > 0:
                self.blink_cooldown -= 1
            
            # Get eye centers
            eye_centers = []
            for (ex, ey, ew, eh) in eyes:
                eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
                center = self._get_pupil_position(eye_roi)
                if center:
                    # Convert to full-frame coordinates
                    eye_centers.append((
                        x + ex + center[0],
                        y + ey + center[1]
                    ))
            
            if eye_centers:
                # Average both eyes
                avg_x = sum(c[0] for c in eye_centers) / len(eye_centers)
                avg_y = sum(c[1] for c in eye_centers) / len(eye_centers)
                
                # Map to screen coordinates
                raw_x = int((avg_x / frame.shape[1]) * WIDTH)
                raw_y = int((avg_y / frame.shape[0]) * HEIGHT)
                
                # Apply dead zone
                raw_x = self._apply_dead_zone(raw_x, WIDTH // 2)
                raw_y = self._apply_dead_zone(raw_y, HEIGHT // 2)
                
                # Apply smoothing
                smooth_x = self._exponential_smooth(raw_x, self.gaze_x)
                smooth_y = self._exponential_smooth(raw_y, self.gaze_y)
                
                self.gaze_x = int(smooth_x)
                self.gaze_y = int(smooth_y)
        
        return True
    
    def get_gaze(self) -> Tuple[int, int]:
        """Get current gaze position."""
        return (self.gaze_x, self.gaze_y)
    
    def is_blinking(self) -> bool:
        """Check if blink detected."""
        return self.blink_detected
    
    def is_face_detected(self) -> bool:
        """Check if face is detected."""
        return self._face_detected
    
    def set_sensitivity(self, sensitivity: float) -> None:
        """Update sensitivity."""
        self.sensitivity = max(0.1, min(2.0, sensitivity))
        self.alpha = 0.3 / self.sensitivity
    
    def get_debug_frame(self) -> Optional[np.ndarray]:
        """Get debug frame."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray, 1.05, 3, minSize=(100, 100)
        )
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 2)
            
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(
                    frame, 
                    (x+ex, y+ey), 
                    (x+ex+ew, y+ey+eh), 
                    (0, 0, 255), 2
                )
                
                eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
                center = self._get_pupil_position(eye_roi)
                if center:
                    pupil_x = x + ex + center[0]
                    pupil_y = y + ey + center[1]
                    cv2.circle(frame, (pupil_x, pupil_y), 5, (255, 0, 0), -1)
        
        # Info text
        face_status = "Face: YES" if self._face_detected else "Face: NO"
        cv2.putText(frame, face_status, (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN if self._face_detected else RED, 2)
        cv2.putText(frame, f"Gaze: ({self.gaze_x}, {self.gaze_y})", (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        cv2.putText(frame, f"Blink: {'YES' if self.blink_detected else 'NO'}", (10, 90), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED if self.blink_detected else WHITE, 2)
        cv2.putText(frame, f"Sens: {self.sensitivity:.1f}", (10, 120), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
        
        return frame
    
    def release(self) -> None:
        """Release resources."""
        if self.cap:
            self.cap.release()


class Fruit:
    """Represents a fruit or bomb."""
    
    FRUIT_TYPES: List[dict] = [
        {"name": "watermelon", "color": GREEN, "score": 10, "emoji": "🍉"},
        {"name": "orange", "color": ORANGE, "score": 15, "emoji": "🍊"},
        {"name": "apple", "color": RED, "score": 20, "emoji": "🍎"},
        {"name": "banana", "color": YELLOW, "score": 25, "emoji": "🍌"},
        {"name": "grape", "color": PURPLE, "score": 30, "emoji": "🍇"},
        {"name": "bomb", "color": BLACK, "score": -50, "emoji": "💣", "is_bomb": True}
    ]
    
    WEIGHTS: List[int] = [20, 20, 20, 20, 15, 5]
    
    def __init__(self, difficulty: int) -> None:
        self.x: float = random.randint(100, WIDTH - 100)
        self.y: float = HEIGHT + 50
        self.vx: float = random.uniform(-4, 4)
        self.vy: float = random.uniform(-18 - difficulty*2, -14 - difficulty)
        self.gravity: float = 0.2
        self.radius: int = 35
        self.sliced: bool = False
        self.slice_time: int = 0
        
        self.type: dict = random.choices(self.FRUIT_TYPES, weights=self.WEIGHTS)[0]
        self.color: Tuple[int, int, int] = self.type["color"]
        self.emoji: str = self.type["emoji"]
        
    def update(self, slow_motion: bool = False) -> None:
        speed_mult = 0.3 if slow_motion else 1.0
        self.x += self.vx * speed_mult
        self.y += self.vy * speed_mult
        self.vy += self.gravity * speed_mult
        if self.sliced:
            self.slice_time += 1
            
    def draw(self, screen: pygame.Surface) -> None:
        if self.sliced and self.slice_time > 10:
            return
            
        font = pygame.font.SysFont(
            "segoe ui emoji, apple color emoji, noto color emoji, sans-serif", 60
        )
        text = font.render(self.emoji, True, WHITE)
        rect = text.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(text, rect)
        
        if self.sliced:
            pygame.draw.line(screen, WHITE, (self.x-30, self.y), (self.x+30, self.y), 3)
            
    def check_slice(self, gaze_pos: Tuple[int, int], slicing: bool) -> bool:
        if self.sliced:
            return False
        dist = math.sqrt(
            (self.x - gaze_pos[0])**2 + (self.y - gaze_pos[1])**2
        )
        if dist < self.radius and slicing:
            self.sliced = True
            return True
        return False
    
    def is_off_screen(self) -> bool:
        return self.y > HEIGHT + 100


class Particle:
    """Visual effect particle."""
    
    def __init__(self, x: float, y: float, color: Tuple[int, int, int]) -> None:
        self.x: float = x
        self.y: float = y
        self.vx: float = random.uniform(-5, 5)
        self.vy: float = random.uniform(-5, 5)
        self.life: int = 30
        self.color: Tuple[int, int, int] = color
        
    def update(self) -> None:
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.2
        self.life -= 1
        
    def draw(self, screen: pygame.Surface) -> None:
        if self.life <= 0:
            return
        alpha = int(255 * (self.life / 30))
        particle_size = 8
        surface = pygame.Surface((particle_size, particle_size), pygame.SRCALPHA)
        pygame.draw.circle(
            surface, (*self.color[:3], alpha), 
            (particle_size // 2, particle_size // 2), particle_size // 2
        )
        screen.blit(
            surface, 
            (int(self.x) - particle_size // 2, int(self.y) - particle_size // 2)
        )


class Game:
    """Main game controller."""
    
    def __init__(self) -> None:
        try:
            self.eye_tracker = EyeTracker(debug=True, sensitivity=1.2)
        except RuntimeError as e:
            print(f"Error: {e}")
            raise
            
        self.state: int = MENU
        self.score: int = 0
        self.lives: int = 3
        self.fruits: List[Fruit] = []
        self.particles: List[Particle] = []
        self.combo: int = 0
        self.max_combo: int = 0
        self.difficulty: int = 1
        self.slow_motion: bool = False
        self.slow_motion_timer: int = 0
        self.ability_ready: bool = False
        
        # Level system
        self.level: int = 1
        self.level_target: int = 200  # Target score for level 1
        
        # Slicing
        self.slicing: bool = False
        self.slice_trail: Deque[Tuple[int, int]] = deque(maxlen=20)
        self.last_blink_time: int = 0
        
        # Gaze trail slicing
        self.slice_cooldown: int = 0  # Frames to wait between slices
        self.slice_cooldown_max: int = 5  # 5 frames cooldown
        
        # Debug mode
        self.debug_mode: bool = False
        
        # Fonts
        self.font_big = pygame.font.Font(None, 74)
        self.font_med = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 36)
        
        # Spawn timing
        self.spawn_timer: int = 0
        self.spawn_delay: int = 60
        
    def spawn_fruit(self) -> None:
        self.fruits.append(Fruit(self.difficulty))
    
    def _line_circle_intersection(
        self,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        circle_center: Tuple[float, float],
        radius: float
    ) -> bool:
        """
        Check if a line segment intersects a circle.
        
        Uses the distance from point to line segment formula.
        If the perpendicular distance from circle center to line segment
        is less than the radius, they intersect.
        
        Args:
            p1: Start point of line segment (x, y)
            p2: End point of line segment (x, y)
            circle_center: Center of circle (x, y)
            radius: Radius of circle
            
        Returns:
            True if line segment intersects circle
        """
        # Vector from p1 to p2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Handle degenerate case (zero-length segment)
        if dx == 0 and dy == 0:
            # Check if point is inside circle
            dist = math.sqrt((p1[0] - circle_center[0])**2 + (p1[1] - circle_center[1])**2)
            return dist <= radius
        
        # Parameter t for closest point on line to circle center
        # t = ((cx - x1) * dx + (cy - y1) * dy) / (dx² + dy²)
        cx, cy = circle_center
        t = ((cx - p1[0]) * dx + (cy - p1[1]) * dy) / (dx * dx + dy * dy)
        
        # Clamp t to segment [0, 1]
        t = max(0, min(1, t))
        
        # Closest point on segment to circle center
        closest_x = p1[0] + t * dx
        closest_y = p1[1] + t * dy
        
        # Distance from closest point to circle center
        dist = math.sqrt((closest_x - cx)**2 + (closest_y - cy)**2)
        
        return dist <= radius
        
    def _check_trail_intersection(self, fruit: Fruit) -> bool:
        """
        Check if any segment in the gaze trail intersects a fruit.
        
        Args:
            fruit: The fruit to check
            
        Returns:
            True if trail intersects fruit hitbox
        """
        if len(self.slice_trail) < 2:
            return False
        
        trail_points = list(self.slice_trail)
        cx, cy = fruit.x, fruit.y
        radius = fruit.radius
        
        # Check each segment in the trail
        for i in range(len(trail_points) - 1):
            p1 = trail_points[i]
            p2 = trail_points[i + 1]
            
            if self._line_circle_intersection(p1, p2, (cx, cy), radius):
                return True
        
        return False
        
    def activate_ability(self) -> None:
        if self.ability_ready:
            self.slow_motion = True
            self.slow_motion_timer = 300
            self.ability_ready = False
            self.combo = 0
            
    def update(self) -> None:
        if self.state != PLAYING:
            return
            
        self.eye_tracker.update()
        gaze = self.eye_tracker.get_gaze()
        
        # Update slice cooldown
        if self.slice_cooldown > 0:
            self.slice_cooldown -= 1
        
        # Always add gaze to trail (for visual effect and intersection checking)
        self.slice_trail.append(gaze)
        
        # Gaze trail slicing - check if any trail segment intersects a fruit
        # No dwell time required - just move your eyes across the fruit!
        self.slicing = False
        
        if self.slice_cooldown == 0:
            for fruit in self.fruits:
                if not fruit.sliced:
                    if self._check_trail_intersection(fruit):
                        self.slicing = True
                        self.slice_cooldown = self.slice_cooldown_max
                        self.last_blink_time = 10  # Visual feedback
                        break
            
            # Also slice on blink - if user blinks while looking at a fruit
            if not self.slicing and self.eye_tracker.is_blinking():
                for fruit in self.fruits:
                    if not fruit.sliced:
                        # Check if gaze is near the fruit
                        dx = gaze[0] - fruit.x
                        dy = gaze[1] - fruit.y
                        distance = math.sqrt(dx * dx + dy * dy)
                        if distance < fruit.radius + 30:  # Fruit radius + some tolerance
                            self.slicing = True
                            self.slice_cooldown = self.slice_cooldown_max
                            self.last_blink_time = 10
                            break
        
        if self.slow_motion:
            self.slow_motion_timer -= 1
            if self.slow_motion_timer <= 0:
                self.slow_motion = False
        
        self.spawn_timer += 1
        current_delay = (
            self.spawn_delay 
            - (self.difficulty * 10) 
            - (self.score // 500)
        )
        current_delay = max(20, current_delay)
        
        if self.spawn_timer >= current_delay:
            self.spawn_fruit()
            self.spawn_timer = 0
            
        fruits_to_remove = []
        for fruit in self.fruits:
            fruit.update(self.slow_motion)
            
            if fruit.check_slice(gaze, self.slicing):
                if fruit.type.get("is_bomb"):
                    self.lives -= 1
                    self.combo = 0
                    self.ability_ready = False
                else:
                    self.score += fruit.type["score"]
                    self.combo += 1
                    
                    if self.combo >= 10 and not self.ability_ready:
                        self.ability_ready = True
                        
                    if self.combo > self.max_combo:
                        self.max_combo = self.combo
                        
                    for _ in range(10):
                        self.particles.append(
                            Particle(fruit.x, fruit.y, fruit.color)
                        )
            
            if fruit.is_off_screen() and not fruit.sliced and not fruit.type.get("is_bomb"):
                self.combo = 0
                self.ability_ready = False
                
            if fruit.is_off_screen() or (fruit.sliced and fruit.slice_time > 30):
                fruits_to_remove.append(fruit)
                
        for fruit in fruits_to_remove:
            if fruit in self.fruits:
                self.fruits.remove(fruit)
        
        for p in self.particles[:]:
            p.update()
            if p.life <= 0:
                self.particles.remove(p)
        
        if self.lives <= 0:
            self.state = GAME_OVER
            
        if self.score >= self.level_target:
            self.state = LEVEL_COMPLETE
            
    def draw(self) -> None:
        if self.slow_motion:
            screen.fill((20, 20, 40))
        else:
            screen.fill(BLACK)
            
        for i in range(0, WIDTH, 50):
            pygame.draw.line(screen, (30, 30, 30), (i, 0), (i, HEIGHT))
        for i in range(0, HEIGHT, 50):
            pygame.draw.line(screen, (30, 30, 30), (0, i), (WIDTH, i))
        
        if self.state == MENU:
            self.draw_menu()
        elif self.state == PLAYING:
            self.draw_game()
            if self.debug_mode:
                self.draw_debug_view()
        elif self.state == PAUSED:
            self.draw_paused()
        elif self.state == LEVEL_COMPLETE:
            self.draw_level_complete()
        elif self.state == GAME_OVER:
            self.draw_game_over()
            
        pygame.display.flip()
        
    def draw_debug_view(self) -> None:
        debug_frame = self.eye_tracker.get_debug_frame()
        if debug_frame is not None:
            debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB)
            debug_frame = np.rot90(debug_frame)
            debug_frame = np.flipud(debug_frame)
            
            debug_height, debug_width = debug_frame.shape[:2]
            max_debug_size = 300
            scale = max_debug_size / max(debug_width, debug_height)
            new_width = int(debug_width * scale)
            new_height = int(debug_height * scale)
            
            debug_surface = pygame.surfarray.make_surface(debug_frame)
            debug_surface = pygame.transform.scale(
                debug_surface, (new_width, new_height)
            )
            
            screen.blit(debug_surface, (10, 10))
            pygame.draw.rect(
                screen, GREEN, (5, 5, new_width + 10, new_height + 10), 2
            )
        
    def draw_menu(self) -> None:
        title = self.font_big.render("EYE NINJA", True, GREEN)
        screen.blit(title, (WIDTH//2 - 200, 200))
        
        subtitle = self.font_med.render("Professional Eye Tracking!", True, WHITE)
        screen.blit(subtitle, (WIDTH//2 - 250, 300))
        
        instructions = [
            "MOVE EYES to aim cursor",
            "BLINK to slice fruits",
            "Slice fruits for points",
            "Avoid bombs!",
            "10 combo = SLOW MOTION"
        ]
        
        y = 400
        for line in instructions:
            text = self.font_small.render(line, True, YELLOW)
            screen.blit(text, (WIDTH//2 - 200, y))
            y += 50
            
        start = self.font_med.render("Press SPACE to Start", True, GREEN)
        screen.blit(start, (WIDTH//2 - 180, 700))
        
    def draw_game(self) -> None:
        gaze = self.eye_tracker.get_gaze()
        
        for p in self.particles:
            p.draw(screen)
        
        for fruit in self.fruits:
            fruit.draw(screen)
        
        if len(self.slice_trail) > 1:
            points = list(self.slice_trail)
            pygame.draw.lines(screen, WHITE, False, points, 4)
            pygame.draw.lines(screen, YELLOW, False, points, 2)
        
        # Draw gaze cursor
        pygame.draw.circle(screen, RED, gaze, 10, 2)
        pygame.draw.circle(screen, RED, gaze, 4)
        pygame.draw.line(screen, RED, (gaze[0]-15, gaze[1]), (gaze[0]+15, gaze[1]), 2)
        pygame.draw.line(screen, RED, (gaze[0], gaze[1]-15), (gaze[0], gaze[1]+15), 2)
        
        # HUD
        score_text = self.font_med.render(f"Score: {self.score}", True, WHITE)
        screen.blit(score_text, (20, 20))
        
        level_text = self.font_small.render(
            f"Level: {self.level} | Target: {self.level_target}", 
            True, GREEN
        )
        screen.blit(level_text, (20, 55))
        
        lives_text = self.font_med.render(f"Lives: {self.lives}", True, RED)
        screen.blit(lives_text, (20, 85))
        
        combo_text = self.font_med.render(f"Combo: {self.combo}", True, YELLOW)
        screen.blit(combo_text, (20, 135))
        
        # Face detection status
        face_detected = self.eye_tracker.is_face_detected()
        face_color = GREEN if face_detected else RED
        face_text = self.font_small.render(
            f"Face: {'Detected' if face_detected else 'Not Found'}",
            True, face_color
        )
        screen.blit(face_text, (20, 175))
        
        # Slice indicator - show when gaze trail is active
        if len(self.slice_trail) > 1:
            slice_text = self.font_small.render("Swipe to slice!", True, GREEN)
        else:
            slice_text = self.font_small.render("Move eyes to slice", True, (100, 100, 100))
        screen.blit(slice_text, (20, 205))
        
        # Ability indicator
        if self.ability_ready:
            ability_text = self.font_big.render(
                "SLOW MOTION READY! (PRESS SPACE)", True, BLUE
            )
            screen.blit(ability_text, (WIDTH//2 - 300, HEIGHT - 100))
        elif self.slow_motion:
            time_left = self.slow_motion_timer // 60 + 1
            ability_text = self.font_big.render(
                f"SLOW MOTION: {time_left}s", True, BLUE
            )
            screen.blit(ability_text, (WIDTH//2 - 200, HEIGHT - 100))
        
        # Debug mode indicator
        debug_color = GREEN if self.debug_mode else (100, 100, 100)
        debug_text = self.font_small.render(
            f"Debug: {'ON' if self.debug_mode else 'OFF'} (Press D)",
            True, debug_color
        )
        screen.blit(debug_text, (WIDTH - 220, 20))
        
    def draw_paused(self) -> None:
        pause_overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pause_overlay.fill((0, 0, 0, 128))
        screen.blit(pause_overlay, (0, 0))
        
        pause_title = self.font_big.render("PAUSED", True, YELLOW)
        title_rect = pause_title.get_rect(center=(WIDTH//2, HEIGHT//2 - 50))
        screen.blit(pause_title, title_rect)
        
        resume_text = self.font_med.render("Press P to Resume", True, WHITE)
        resume_rect = resume_text.get_rect(center=(WIDTH//2, HEIGHT//2 + 20))
        screen.blit(resume_text, resume_rect)
        
    def draw_level_complete(self) -> None:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        screen.blit(overlay, (0, 0))
        
        level_text = self.font_big.render(f"LEVEL {self.level} COMPLETE!", True, GREEN)
        screen.blit(level_text, (WIDTH//2 - 200, 250))
        
        score_text = self.font_med.render(f"Score: {self.score}", True, WHITE)
        screen.blit(score_text, (WIDTH//2 - 100, 350))
        
        next_target = self.level_target + 500
        target_text = self.font_med.render(
            f"Next Level Target: {next_target}", True, YELLOW
        )
        screen.blit(target_text, (WIDTH//2 - 200, 420))
        
        continue_text = self.font_med.render(
            "Press SPACE for Next Level", True, GREEN
        )
        screen.blit(continue_text, (WIDTH//2 - 200, 520))
        
    def draw_game_over(self) -> None:
        over_text = self.font_big.render("GAME OVER", True, RED)
        screen.blit(over_text, (WIDTH//2 - 150, 300))
        
        final_score = self.font_med.render(
            f"Final Score: {self.score}", True, WHITE
        )
        screen.blit(final_score, (WIDTH//2 - 150, 400))
        
        max_combo_text = self.font_med.render(
            f"Max Combo: {self.max_combo}", True, YELLOW
        )
        screen.blit(max_combo_text, (WIDTH//2 - 150, 460))
        
        restart = self.font_med.render(
            "Press R to Restart or Q to Quit", True, GREEN
        )
        screen.blit(restart, (WIDTH//2 - 250, 550))
        
    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return False
                    
                if self.state == MENU:
                    if event.key == pygame.K_SPACE:
                        self.state = PLAYING
                        
                elif self.state == PLAYING:
                    if event.key == pygame.K_SPACE and self.ability_ready:
                        self.activate_ability()
                    elif event.key == pygame.K_p:
                        self.state = PAUSED
                    elif event.key == pygame.K_ESCAPE:
                        self.state = MENU
                    elif event.key == pygame.K_d:
                        self.debug_mode = not self.debug_mode
                    elif event.key == pygame.K_UP:
                        self.eye_tracker.set_sensitivity(
                            self.eye_tracker.sensitivity + 0.2
                        )
                    elif event.key == pygame.K_DOWN:
                        self.eye_tracker.set_sensitivity(
                            self.eye_tracker.sensitivity - 0.2
                        )
                        
                elif self.state == PAUSED:
                    if event.key == pygame.K_p:
                        self.state = PLAYING
                    elif event.key == pygame.K_ESCAPE:
                        self.state = MENU
                        
                elif self.state == LEVEL_COMPLETE:
                    if event.key == pygame.K_SPACE:
                        self.next_level()
                    elif event.key == pygame.K_ESCAPE:
                        self.state = MENU
                        
                elif self.state == GAME_OVER:
                    if event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_q:
                        return False
                        
        return True
    
    def next_level(self) -> None:
        self.level += 1
        self.level_target += 500
        self.difficulty = min(3, 1 + (self.level - 1) // 2)
        self.state = PLAYING
    
    def reset_game(self) -> None:
        self.score = 0
        self.lives = 3
        self.fruits = []
        self.particles = []
        self.combo = 0
        self.max_combo = 0
        self.slow_motion = False
        self.ability_ready = False
        self.level = 1
        self.level_target = 500
        self.difficulty = 1
        self.state = PLAYING
        
    def run(self) -> None:
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            clock.tick(60)
            
        self.eye_tracker.release()
        pygame.quit()


def main() -> None:
    print("EYE NINJA - Professional Eye Tracking")
    print("=" * 40)
    print("Controls:")
    print("  - Move EYES to aim")
    print("  - BLINK to slice")
    print("  - Z to slice manually")
    print("  - D for debug camera view")
    print("  - UP/DOWN to adjust sensitivity")
    print("  - SPACE to activate ability")
    print("  - P to pause")
    print()
    print("Tip: Make sure your face is visible to the camera!")
    print("Press SPACE to start playing!")
    
    try:
        game = Game()
        game.run()
    except Exception as e:
        print(f"Error: {e}")
        pygame.quit()


if __name__ == "__main__":
    main()
