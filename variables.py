

familiarity = [
    'Q9'
]

kms_driven = [
    'Q2'
]

####################
### DEMOGRAPHICS ###
####################

demographics = [
    "gender",
    "colour_plus",
    "region",
    "Q20", # How many people (adults and children) live in your household?
    "Q21", # What kind of home do you live in?
    "Q22", # Do you own your home or rent it?
    "AgeBand",
    "Q24" # What is your approximate household income?
    ]


##################
### CAPABILITY ###
##################

input_variables_vehicle_ownership = [
    'Q1_1',
    'Q1_2',
    'Q1_3',
    'Q1_4',
    'Q1_99'
]

parking =  [
    "Q3_1",
    "Q3_2",
    "Q3_3",
    "Q3_4",
    "Q3_5"
]

##############
### HABITS ###
##############

# Define Q6 (weekdays) columns by vehicle + time
# "Q6ax1" => weekdays, x1 => petrol/diesel
# "Q6ax2" => weekdays, x2 => electric
# "Q6ax3" => weekdays, x3 => plug-in hybrid
# "Q6ax4" => weekdays, x4 => hybrid

# Each has 3 time periods: 
# _1 => 10am-3pm, 
# _2 => 4pm-9pm, 
# _3 => 9pm-6am

q6a_petrol = ["Q6ax1_1","Q6ax1_2","Q6ax1_3"]       # weekdays, petrol/diesel
q6a_ev     = ["Q6ax2_1","Q6ax2_2","Q6ax2_3"]       # weekdays, EV
q6a_plughyb= ["Q6ax3_1","Q6ax3_2","Q6ax3_3"]       # weekdays, plug-in hybrid
q6a_hybrid = ["Q6ax4_1","Q6ax4_2","Q6ax4_3"]       # weekdays, hybrid

# Combine them all for weekdays
q6a_cols = q6a_petrol + q6a_ev + q6a_plughyb + q6a_hybrid

energ_literacy = [
    "Q7_1",
    "Q7_2",
    "Q7_3",
]

renewables_at_home = [
    'Q8_1',
    'Q8_2',
    'Q8_99'
]

intent_to_purchase_vehicle = [
    "Q5_1",
    "Q5_2",
    "Q5_3",
    "Q5_4",
    "Q5_99"
]

################
### ATTIUDES ###
################

benefits_v2g = [
    "Q14_1",
    "Q14_2",
    "Q14_3",
    "Q14_4",
    "Q14_5",
    "Q14_6",
    "Q14_7",
    "Q14_8",
#    "Q14_98",
    "Q14_99"
]

concerns_v2g = [
    "Q15_1",
    "Q15_2",
    "Q15_3",
    "Q15_4",
    "Q15_5",
    "Q15_6",
    "Q15_7",
    "Q15_8",
    "Q15_9",
    "Q15_10",
#    "Q15_98",
    "Q15_99",
]

interest_in_services = [
    "Q16_1",
    "Q16_2",
    "Q16_3",
    "Q16_4",
    "Q16_5",
    "Q16_6",
    "Q16_7",
    "Q16_8",
    "Q16_9"
]

consider_using_NRMA_for = [
    "Q18_1",
    "Q18_2",
    "Q18_3",
    "Q18_4",
    "Q18_5",
    "Q18_6",
    "Q18_7",
    "Q18_8",
    "Q18_9",
    "Q18_99"
]

charging_control = [
    "Q17_1",
    "Q17_2",
    "Q17_3",
]

preference_on_batt_use = [
    "Q18"
]

expected_return = [
    "Q19"
]

######################
### FEATURE LABELS ###
######################

feature_label_map = {
    "Q1_1": "Owns Petrol/Diesel Car",
    "Q1_2": "Owns Electric Vehicle",
    "Q1_3": "Owns Plug-in Hybrid",
    "Q1_4": "Owns Hybrid",
    "Q1_99": "Does Not Own a Car",
    "Q3_1": "Personal driveway",
    "Q3_2": "Personal garage",
    "Q3_3": "Carport",
    "Q3_4": "Street parking",
    "Q3_5": "Other parking",
    "Q9": "How Familiar Are You with V2G?",
    "Q2": "How Many Kilometres Do You Drive Per Year?",
    "Q8_1": "Owns Solar Panels",
    "Q8_2": "Owns Home Battery",
    "Q8_99": "Owns Neither Solar nor Battery",

    "Q6ax1_1":"Petrol at home (10am-3pm)", 
    "Q6ax1_2":"Petrol at home (4pm-9pm)", 
    "Q6ax1_3":"Petrol at home (9pm-6am)",

    "Q6ax2_1":"EV at home (10am-3pm)", 
    "Q6ax2_2":"EV at home (4pm-9pm)", 
    "Q6ax2_3":"EV at home (9pm-6am)",

    "Q6ax3_1":"Plug-in Hybrid at home (10am-3pm)", 
    "Q6ax3_2":"Plug-in Hybrid at home (4pm-9pm)", 
    "Q6ax3_3":"Plug-in Hybrid at home (9pm-6am)",

    "Q6ax4_1":"Hybrid at home (10am-3pm)", 
    "Q6ax4_2":"Hybrid at home (4pm-9pm)", 
    "Q6ax4_3":"Hybrid at home (9pm-6am)",

    "Q7_1":"I check my energy bill regularly",
    "Q7_2":"I try to minimise my energy usage to reduce my bill",
    "Q7_3":"I regularly compare my current plan against alternatives in the market",

    "Q14_1": "Earning money from my car battery by selling electricity to the grid",
    "Q14_2": "Saving on my energy bill by using my car battery to power my home",
    "Q14_3": "Helping the environment",
    "Q14_4": "Using more of my solar power",
    "Q14_5": "Supporting the grid during high demand",
    "Q14_6": "Being more independent from the grid",
    "Q14_7": "Protecting me against blackout",
    "Q14_8": "Avoiding buying a home battery",
    "Q14_99": "Q14 - None of these",

    "Q15_1": "High cost of car and charger",
    "Q15_2": "Safety concerns",
    "Q15_3": "Complicated to set up",
    "Q15_4": "Complicated to use",
    "Q15_5": "Impact on my EV battery life and range",
    "Q15_6": "Not having enough battery when I need to drive",
    "Q15_7": "Impact on my EV warranty",
    "Q15_8": "Whether it is actually possible at my home",
    "Q15_9": "Would require my car to be plugged in for the majority of the time",
    "Q15_10": "I don’t want to lose control of how I charge my EV",
    "Q15_99": "Q15 - None of these",

    "Q17_1": "I manually select when I want to charge my car and when I want to discharge my car to sell energy back to the grid",
    "Q17_2": "I use software to automatically decide whether to charge the car or to sell energy back to the grid, based on earning the most money (within agreed limits that do not impact my driving requirements)",
    "Q17_3": "I have an agreement with a company that they can discharge half my car battery once a week and they provide me with a regular payment"

}