from abc import ABC, abstractmethod

class VideoSource(ABC):
   @abstractmethod
   def get_video_by_keyword(self, keyword, amount=1):
       pass