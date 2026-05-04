"""
Django settings for project_settings project.
"""

import os


def load_env_file(path):
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# When running as a frozen EXE, DEEPFAKE_PROJECT_DIR points to the EXE directory
# so that models/ and .env are found next to the executable.
PROJECT_DIR = os.environ.get(
    "DEEPFAKE_PROJECT_DIR",
    os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
REPO_ROOT = os.path.dirname(PROJECT_DIR)

# In frozen EXE mode, only load .env from next to the EXE (PROJECT_DIR).
# In source mode, also check the repo root (parent of Django Application).
if not os.environ.get("DEEPFAKE_PROJECT_DIR"):
    load_env_file(os.path.join(REPO_ROOT, '.env'))
load_env_file(os.path.join(PROJECT_DIR, '.env'))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/3.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = '@)0qp0!&-vht7k0wyuihr+nk-b8zrvb5j^1d@vl84cd1%)f=dz'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

# Change and set this to correct IP/Domain
ALLOWED_HOSTS = ["*"]


# Application definition

INSTALLED_APPS = [
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'ml_app.apps.MlAppConfig'
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'project_settings.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(PROJECT_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.messages.context_processors.messages',
                'django.template.context_processors.media'
            ],
        },
    },
]

WSGI_APPLICATION = 'project_settings.wsgi.application'


# Database
# https://docs.djangoproject.com/en/3.0/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": os.path.join(PROJECT_DIR, 'db.sqlite3'),
    }
}


# Internationalization
# https://docs.djangoproject.com/en/3.0/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = False

USE_L10N = False

USE_TZ = False


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/3.0/howto/static-files/

#used in production to serve static files
STATIC_ROOT = "/home/app/staticfiles/"

#url for static files
STATIC_URL = '/static/'

STATICFILES_DIRS = [
    os.path.join(PROJECT_DIR, 'uploaded_images'),
    os.path.join(PROJECT_DIR, 'static'),
    os.path.join(PROJECT_DIR, 'models'),
]

DATA_UPLOAD_MAX_MEMORY_SIZE = None
FILE_UPLOAD_MAX_MEMORY_SIZE = 0

MEDIA_URL = "/media/"

MEDIA_ROOT = os.path.join(PROJECT_DIR, 'uploaded_videos')

ENABLE_GEMINI_REVIEW = os.getenv('ENABLE_GEMINI_REVIEW', 'true').strip().lower() in ('1', 'true', 'yes', 'on')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-flash-lite-latest')
GEMINI_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv(
        'GEMINI_FALLBACK_MODELS',
        'gemini-3-flash-preview,gemini-2.5-flash,gemini-2.0-flash,gemini-2.0-flash-lite,gemini-flash-latest,gemini-flash-lite-latest',
    ).split(',')
    if model.strip()
]
LOCAL_FAKE_THRESHOLD = float(os.getenv('LOCAL_FAKE_THRESHOLD', '0.40'))
VIDEO_LOCAL_FAKE_THRESHOLD = float(os.getenv('VIDEO_LOCAL_FAKE_THRESHOLD', '0.55'))
GEMINI_REVIEW_BAND_LOW = float(os.getenv('GEMINI_REVIEW_BAND_LOW', '0.20'))
GEMINI_REVIEW_BAND_HIGH = float(os.getenv('GEMINI_REVIEW_BAND_HIGH', '0.40'))
GEMINI_TIMEOUT_SECONDS = float(os.getenv('GEMINI_TIMEOUT_SECONDS', '20'))
GEMINI_REAL_OVERRIDE_MIN_CONFIDENCE = float(os.getenv('GEMINI_REAL_OVERRIDE_MIN_CONFIDENCE', '70'))
GEMINI_REAL_OVERRIDE_MAX_FAKE_PROB = float(os.getenv('GEMINI_REAL_OVERRIDE_MAX_FAKE_PROB', '0.70'))

#for extra logging in production environment
if DEBUG == False:
    LOGGING = {
        'version': 1,
        'disable_existing_loggers': False,
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
            },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': 'log.django',
        },
        },
        'loggers': {
            'django': {
                'handlers': ['console','file'],
                'level': os.getenv('DJANGO_LOG_LEVEL', 'DEBUG'),
            },
        },
    }
