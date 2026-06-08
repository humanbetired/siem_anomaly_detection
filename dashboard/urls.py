from django.urls import path
from dashboard import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/ingest/', views.ingest_flow, name='ingest'),
    path('api/stats/', views.get_stats, name='stats'),
]