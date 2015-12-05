from django.conf.urls import patterns, include, url
from django.contrib import admin

from estimators import views

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'sk_analytics.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^admin/', include(admin.site.urls)),
    url(r'^estimators/train', views.train),
    url(r'^estimators/rebuild_model', views.rebuild_model),
    url(r'^estimators/predict', views.predict),
)
