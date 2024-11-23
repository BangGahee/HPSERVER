from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # index 뷰
    path('chat/', views.chat, name='chat'),  # chat 뷰
]
