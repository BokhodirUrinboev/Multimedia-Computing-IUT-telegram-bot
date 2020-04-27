##############################################################################################################
import os
from flask import Flask, request
import telebot
from telebot import types

###############################################################################################################
photos_path ="C:\\Users\\Bokhodir\\PycharmProjects\\Multimedia-Computing-IUT-telegram-bot\\static\\Photos\\"
API_TOKEN = '1087119303:AAGcKmpW5_FN4elBJ4M9O1wfeDkDw8gY1uM'

###############################################################################################################
bot = telebot.TeleBot(API_TOKEN)
app = Flask(__name__)

###############################################################################################################
USERS = []
DOCUMENT_SCAN_BUTTON = "SCAN DOCUMENT"
SKETCH_BUTTON = "SKETCH IMAGE"
###############################################################################################################


@bot.message_handler(commands=['start'])
def start(message):
    global USERS
    chat_id = message.chat.id
    for i in USERS:
        if i['chat_id'] == chat_id:
            USERS.remove(i)

    markup = types.ReplyKeyboardMarkup()

    markup.add(types.KeyboardButton(DOCUMENT_SCAN_BUTTON))
    markup.add(types.KeyboardButton(SKETCH_BUTTON))
    bot.send_message(chat_id=chat_id, text="CHOOSE OPERATION TO PERFORM", reply_markup=markup)


def button_checker_dsb(message):
    return message.text == DOCUMENT_SCAN_BUTTON


def button_checker_sb(message):
    return message.text == DOCUMENT_SCAN_BUTTON


#####################################scan_document###################################################


@bot.message_handler(func=button_checker_dsb)
def document_scan_msg(message):
    global USERS
    chat_id = message.chat.id
    USER = {}
    USER['chat_id'] = chat_id
    USERS.append(USER)
    msg = bot.send_message(chat_id=chat_id, text="Now send me photo of document to scan")
    bot.register_next_step_handler(msg, document_scan)


def document_scan(message):
    global USERS
    msg = message.text
    chat_id = message.chat.id
    USER = {}
    for i in USERS:
        if i['chat_id'] == chat_id:
            USER = i
    msg = bot.send_message(chat_id=chat_id, text="You choosed document scan operation")
    bot.register_next_step_handler(msg, start)

##############################################sketch_photo##########################################################


@bot.message_handler(func=button_checker_sb)
def sketch_photo_msg(message):
    global USERS
    chat_id = message.chat.id
    USER = {}
    USER['chat_id'] = chat_id
    USERS.append(USER)
    msg = bot.send_message(chat_id=chat_id, text="Now send me photo to sketch")
    bot.register_next_step_handler(msg, document_scan)


def sketch_photo(message):
    global USERS
    msg = message.text
    chat_id = message.chat.id
    USER = {}
    for i in USERS:
        if i['chat_id'] == chat_id:
            USER = i
    msg = bot.send_message(chat_id=chat_id, text="You choosed sket photo")
    bot.register_next_step_handler(msg, start)


################################################################################################################

bot.polling()
