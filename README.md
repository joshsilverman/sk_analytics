# sk Analytics


## Install python 2.7

* On Mac: download and run installer from "https://www.python.org/downloads/"
* On Ubuntu: python 2.7 should already be installed and fine to use. If not, download source and run "./configure ; make ; sudo make install"

## Install pip

* On Mac, the 2.7.9 python install comes with pip.
* On Ubuntu, "sudo apt-get install python-pip".
* Either way, you may need to upgrade it with "[sudo] pip install --upgrade pip"
* Also, on Ubuntu you may need to also "sudo apt-get install python-dev" to get some of the build dependencies installing the requirements.

## Run "[sudo] pip install -r requirements.txt". 
NOTE: in order to get this working on Ubuntu you may have to install the following as well (some may be installed already):

* sudo apt-get install liblapack-dev
* sudo apt-get install gfortran
* sudo apt-get libblas-dev

## Create database in local postgres

* Create "sk_analytics" database user. In psql:
```
#!SQL

CREATE USER sk_analytics WITH SUPERUSER ;

```

* Create "sk_analytics" database:
```
#!SQL

CREATE DATABASE sk_analytics WITH OWNER = sk_analytics ;

```

* Update the pg_hba.conf file (typically in /etc/postgresql/9.4/main/pg_hba.conf) to allow password-free authentication on localhost for the sk_analytics user. Add the following line (using a host rule instead of a local rule is important here - the local rule, as we do with rails, does not seem compatible with the connection parameters we're passing in):

```
#!python

host    all sk_analytics 127.0.0.1/32  trust

```

## Run "python ./manage.py migrate"
Should show no errors / failures.

## Run "python ./manage.py test"
All tests should pass.

## Example training session:

```
#!sh

$ python manage.py runserver

# In a second shell window:
$ export sk_ANALYTICS_BASE_URL='http://127.0.0.1:8000'
$ rake train_predict:train[48,win_loss,sgdclassifier]

```