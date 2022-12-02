import pygame

class Renderer:
    font = None
    screen = None
    
    def __init__(self, screen_size, font_size=16):
        pygame.init()
        self.font_size = font_size
        self.font = pygame.font.SysFont("dejavusans", self.font_size)
        self.screen_size = screen_size
        self.screen = pygame.display.set_mode(self.screen_size)
    
    def quit(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
    
    def show(self, bitmap):
        self.screen.blit(bitmap, (0, 0))
        pygame.display.flip()
    
    def add_global_info(self, bitmap, info):
        back = pygame.Surface((128, 32))
        pygame.draw.rect(back, (80, 0, 0, 128), (0, 0, 128, 32))
        for i, text in enumerate(info):
            foreground = self.font.render(text, True, (255, 255, 255, 100))
            back.blit(foreground, (0, i * 16))
        bitmap.blit(back, ((self.screen_size[0] // 2) - (128//2), (self.screen_size[1] // 2) - (self.font_size * len(info))), None, pygame.BLEND_RGBA_MULT)
    
    def add_field_info(self, bitmap, field, info):
        for i, text in enumerate(info):
            foreground = self.font.render(text, True, (0, 0, 0))
            bitmap.blit(foreground, (field.render_info["start_x"], field.render_info["start_y"] + i * self.font_size))
    
    def get_surface(self, color, transparency=False):
        surface = pygame.Surface(self.screen_size, pygame.SRCALPHA)
        if transparency:
            surface.convert_alpha()
        surface.fill(color)
        return surface