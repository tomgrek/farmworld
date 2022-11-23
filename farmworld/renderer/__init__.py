import pygame

class Renderer:
    font = None
    screen = None
    
    def __init__(self, screen_size):
        pygame.init()
        self.font = pygame.font.SysFont("dejavusans", 16)
        self.screen_size = screen_size
        self.screen = pygame.display.set_mode(self.screen_size)
    
    def quit(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
    
    def show(self, bitmap):
        self.screen.blit(bitmap, (0, 0))
        pygame.display.flip()
    
    def add_info(self, bitmap, info):
        for i, text in enumerate(info):
            foreground = self.font.render(text, True, (0, 0, 0))
            bitmap.blit(foreground, (0, i * 16))
    
    def get_surface(self, color, transparency=False):
        surface = pygame.Surface(self.screen_size, pygame.SRCALPHA)
        if transparency:
            surface.convert_alpha()
        surface.fill(color)
        return surface