from rasa_sdk import Action
from rasa_sdk.events import SlotSet


class RestaurantAPI(object):
    def search(self, info):
        return "金色三麥台中市政店"


class ActionSearchRestaurants(Action):
    def name(self):
        return "action_search_restaurants"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message("正在找尋餐廳")
        restaurant_api = RestaurantAPI()
        restaurants = restaurant_api.search(tracker.get_slot("cuisine"))
        return [SlotSet("matches", restaurants)]


class ActionSuggest(Action):
    def name(self):
        return "action_suggest"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message("這是我找到的餐廳:")
        dispatcher.utter_message(tracker.get_slot("matches"))
        dispatcher.utter_message(
            "還行嗎？暗示: 你快點滾吧 :)"
        )
        return []
