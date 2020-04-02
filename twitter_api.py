import time
import constants
from twython import Twython


def get_user_twitter():
    return Twython(constants.APP_KEY, constants.APP_SECRET, constants.OAUTH_TOKEN, constants.OAUTH_TOKEN_SECRET)

def get_app_twitter():
    twitter = Twython(constants.APP_KEY, constants.APP_SECRET, oauth_version=2)
    ACCESS_TOKEN = twitter.obtain_access_token()

    twitter = Twython(constants.APP_KEY, access_token=ACCESS_TOKEN)

    return twitter

def check_rate(twitter, category, endpoint):
    result = twitter.get_application_rate_limit_status()
    return result['resources'][category][endpoint]['remaining']


class Budget():
    def __init__(self, initial, first_round=None):
        self._initial = initial
        if first_round:
            self._current = first_round
        else:
            self._current = initial
    
    def use(self):
        if self._current > 0:
            self._current -= 1
            return True
        else:
            return False
    
    def reset(self):
        self._current = self._initial


def make_budgets(user_val, app_val, category, endpoint):
    user_twitter = get_user_twitter()
    app_twitter = get_app_twitter()
    user_remaining = check_rate(user_twitter, category, endpoint)
    app_remaining = check_rate(app_twitter, category, endpoint)
    return [Budget(user_val, first_round=user_remaining), Budget(app_val, first_round=app_remaining)]


class TwitterInterface():
    def __init__(self, budgets):
        self.user_twitter = get_user_twitter()
        self.app_twitter = get_app_twitter()
        self.budgets = budgets
        self.counter = 0

    def get_connector(self):
        self.counter += 1
        if self.budgets[0].use():
            return self.user_twitter
        elif self.budgets[1].use():
            return self.app_twitter
        else:
            print('sleeping')
            time.sleep((60 * 15) + 1)
            self.app_twitter = get_app_twitter() # It won't hurt to get a new token
            self.budgets[0].reset()
            self.budgets[1].reset()

            self.budgets[0].use()
            return self.user_twitter
