
from django.contrib import admin
from django.urls import path
import myapp.views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', myapp.views.home, name='home'),
    path('add/', myapp.views.add, name='add'),
    path('pred/', myapp.views.pred, name='pred')
]
