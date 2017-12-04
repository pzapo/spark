# UDF FUNCTIONS
def profit_(today_price, previous_day_price):
    if today_price - previous_day_price > 0:
        return 1
    else:
        return 0
