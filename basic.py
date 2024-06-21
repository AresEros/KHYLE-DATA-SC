'''print('hello wold')
print ('hollo')'''

#Basic 
'''phrase = "khyle"
print (phrase.upper())'''


'''name = input ('Enter your name ' )
password = "miss"
print ('Hello ' + name)
pas = input ('please Enter your Password ' )

if pas != password :
    print ('wrong passworld')
else:
    print ('welcome ' + name )'''


'''is_male = True
is_tall = False

if is_male and is_tall:
    print ("you are a tall male")
elif is_male and not(is_tall):
    print ('you are a short male')
elif not(is_male) and is_tall:
    print('you are not a male but tall')
else:
    print ('you are either not male or not tall or both')'''

# IF STATEMENT AND COMPERISON

'''def max_num(num1, num2, num3):
    if num1 >= num2 and num1 >= num3:
        return num1
    elif num2 >= num1 and num3:
        return num2
    else:
        return num3
print(max_num(30, 40, 8))'''


#DICTIONARY
month_conversions = {
    'jan' : 'january',
    'fab' : 'febuary',
    'mar' : 'march',
    'apr' : 'april',
    'may' : 'may',
    'jun' : 'june',
    'jul' : 'july',
    'aug' : 'august',
    'sep' : 'september',
    'oct' : 'october',
    "nov" : 'november',
    'dec' : 'december',
}

print (month_conversions.get('mar'))