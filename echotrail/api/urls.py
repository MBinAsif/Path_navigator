from django.urls import path
from .views import ObstacleDetectionAPIView

urlpatterns = [
    path('obstacle-detect/', ObstacleDetectionAPIView.as_view(), name='obstacle_detect'),
]
