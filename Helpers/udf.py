# UDF FUNCTIONS
def profit_(today_price, previous_day_price):
    if today_price - previous_day_price > 0:
        return 1
    else:
        return 0


def ReverseTradeClassifier(profit):
    return 1.0 if profit == 0.0 else 0.0


def BuyAndHoldClassifier(yesterday, today):
    if yesterday and today == 1.0:
        return 1.0
    else:
        return 0
