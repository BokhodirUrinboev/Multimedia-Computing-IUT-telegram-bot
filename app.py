##############################################################################################################
import os
from flask import Flask, request
import telebot
import cv2
from telebot import types
import datetime
from opencv_app import convert_to_pencil_sketch, scaner, cartoonize_photos, vignette_filter_photo
from setings import API_TOKEN

###############################################################################################################
photos_path ="C:\\Users\\Bokhodir\\PycharmProjects\\Multimedia-Computing-IUT-telegram-bot\\static\\Photos\\"

app = Flask(__name__)
###############################################################################################################
bot = telebot.TeleBot(API_TOKEN)
###############################################################################################################
USERS = []
DOCUMENT_SCAN_BUTTON = "SCAN DOCUMENT"
SKETCH_BUTTON = "SKETCH IMAGE"
CARTUNIZE_BUTTON = "CARTOONIZE IMAGE"
VIGNETTE_FILTER_BUTTON = "VIGNETTE FILTER IMAGE"
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
    markup.add(types.KeyboardButton(CARTUNIZE_BUTTON))
    markup.add(types.KeyboardButton(VIGNETTE_FILTER_BUTTON))

    bot.send_message(chat_id=chat_id, text="CHOOSE OPERATION TO PERFORM", reply_markup=markup)

###############################################################################################
def button_checker_dsb(message):
    return message.text == DOCUMENT_SCAN_BUTTON

def button_checker_sb(message):
    return message.text == SKETCH_BUTTON

def button_checker_cb(message):
    return message.text == CARTUNIZE_BUTTON

def button_checker_vfb(message):
    return message.text == VIGNETTE_FILTER_BUTTON

##################################################################################################
#################################### scan_document ###################################################


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
    global photos_path

    msg = message.text
    chat_id = message.chat.id
    if message.content_type == 'photo':
        USER = {}
        for i in USERS:
            if i['chat_id'] == chat_id:
                USER = i
                basename = photos_path + "user_photo"
                suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S" + ".jpg")
                photo_name = "_".join([basename, suffix])
                USER['photo_name'] = photo_name
        file_info = message.photo[-1].file_id
        file = bot.get_file(file_info)
        downloaded_file = bot.download_file(file.file_path)

        with open(USER['photo_name'], 'wb') as new_file:
            new_file.write(downloaded_file)
        image = cv2.imread(USER['photo_name'])
        scaned = scaner(image)
        cv2.imwrite(USER['photo_name'], scaned)
        photo = open(USER['photo_name'], 'rb')
        msg = bot.send_photo(USER['chat_id'], photo)
        #os.remove(USER['photo_name'])
        for i in USERS:
            if i['chat_id'] == chat_id:
                USERS.remove(i)

        markup = types.ReplyKeyboardMarkup()

        markup.add(types.KeyboardButton(DOCUMENT_SCAN_BUTTON))
        markup.add(types.KeyboardButton(SKETCH_BUTTON))
        markup.add(types.KeyboardButton(CARTUNIZE_BUTTON))
        markup.add(types.KeyboardButton(VIGNETTE_FILTER_BUTTON))

        bot.send_message(chat_id=chat_id, text="CHOOSE OPERATION TO PERFORM", reply_markup=markup)
    else:
        msg = bot.send_message(chat_id=chat_id, text="Send Photo only")
        bot.register_next_step_handler(msg, document_scan)


##############################################sketch_photo##########################################################


@bot.message_handler(func=button_checker_sb)
def sketch_photo_msg(message):
    global USERS
    chat_id = message.chat.id
    USER = {}
    USER['chat_id'] = chat_id
    USERS.append(USER)
    msg = bot.send_message(chat_id=chat_id, text="Now send me photo to sketch")
    bot.register_next_step_handler(msg, sketch_photo)


def sketch_photo(message):
    global USERS
    global photos_path

    msg = message.text
    chat_id = message.chat.id
    USER = {}
    if message.content_type == 'photo':
        for i in USERS:
            if i['chat_id'] == chat_id:
                USER = i
                basename = photos_path + "user_photo"
                suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S" + ".jpg")
                photo_name = "_".join([basename, suffix])
                USER['photo_name'] = photo_name
        file_info = message.photo[-1].file_id
        file = bot.get_file(file_info)
        downloaded_file = bot.download_file(file.file_path)

        with open(USER['photo_name'], 'wb') as new_file:
            new_file.write(downloaded_file)
        image = cv2.imread(USER['photo_name'])
        sketched = convert_to_pencil_sketch(image)
        cv2.imwrite(USER['photo_name'], sketched)
        photo = open(USER['photo_name'], 'rb')
        msg = bot.send_photo(USER['chat_id'], photo)
        #os.remove(USER['photo_name'])
        for i in USERS:
            if i['chat_id'] == chat_id:
                USERS.remove(i)

        markup = types.ReplyKeyboardMarkup()

        markup.add(types.KeyboardButton(DOCUMENT_SCAN_BUTTON))
        markup.add(types.KeyboardButton(SKETCH_BUTTON))
        markup.add(types.KeyboardButton(CARTUNIZE_BUTTON))
        markup.add(types.KeyboardButton(VIGNETTE_FILTER_BUTTON))

        bot.send_message(chat_id=chat_id, text="CHOOSE OPERATION TO PERFORM", reply_markup=markup)
    else:
        msg = bot.send_message(chat_id=chat_id, text="Send Photo only")
        bot.register_next_step_handler(msg, sketch_photo)


################################################################################################################


##############################################crtoonize_photo##########################################################


@bot.message_handler(func=button_checker_cb)
def cartoonize_photo_msg(message):
    global USERS
    chat_id = message.chat.id
    USER = {}
    USER['chat_id'] = chat_id
    USERS.append(USER)
    msg = bot.send_message(chat_id=chat_id, text="Now send me photo of yours in order to make it funny")
    bot.register_next_step_handler(msg, cartoonize_photo)


def cartoonize_photo(message):
    global USERS
    global photos_path

    msg = message.text
    chat_id = message.chat.id
    USER = {}
    if message.content_type == 'photo':
        for i in USERS:
            if i['chat_id'] == chat_id:
                USER = i
                basename = photos_path + "user_photo"
                suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S" + ".jpg")
                photo_name = "_".join([basename, suffix])
                USER['photo_name'] = photo_name
        file_info = message.photo[-1].file_id
        file = bot.get_file(file_info)
        downloaded_file = bot.download_file(file.file_path)

        with open(USER['photo_name'], 'wb') as new_file:
            new_file.write(downloaded_file)
        image = cv2.imread(USER['photo_name'])
        sketched = cartoonize_photos(image)
        cv2.imwrite(USER['photo_name'], sketched)
        photo = open(USER['photo_name'], 'rb')
        msg = bot.send_photo(USER['chat_id'], photo)
        #os.remove(USER['photo_name'])
        for i in USERS:
            if i['chat_id'] == chat_id:
                USERS.remove(i)

        markup = types.ReplyKeyboardMarkup()

        markup.add(types.KeyboardButton(DOCUMENT_SCAN_BUTTON))
        markup.add(types.KeyboardButton(SKETCH_BUTTON))
        markup.add(types.KeyboardButton(CARTUNIZE_BUTTON))
        markup.add(types.KeyboardButton(VIGNETTE_FILTER_BUTTON))
        bot.send_message(chat_id=chat_id, text="CHOOSE OPERATION TO PERFORM", reply_markup=markup)
    else:
        msg = bot.send_message(chat_id=chat_id, text="Send Photo only")
        bot.register_next_step_handler(msg, cartoonize_photo)


################################################################################################################


#################################### vignette filter ###################################################


@bot.message_handler(func=button_checker_vfb)
def vignette_filter_msg(message):
    global USERS
    chat_id = message.chat.id
    USER = {}
    USER['chat_id'] = chat_id
    USERS.append(USER)
    msg = bot.send_message(chat_id=chat_id, text="Now send me photo to filter it by nice vignette filter.")
    bot.register_next_step_handler(msg, vignette_filter)


def vignette_filter(message):
    global USERS
    global photos_path

    msg = message.text
    chat_id = message.chat.id
    if message.content_type == 'photo':
        USER = {}
        for i in USERS:
            if i['chat_id'] == chat_id:
                USER = i
                basename = photos_path + "user_photo"
                suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S" + ".jpg")
                photo_name = "_".join([basename, suffix])
                USER['photo_name'] = photo_name
        file_info = message.photo[-1].file_id
        file = bot.get_file(file_info)
        downloaded_file = bot.download_file(file.file_path)

        with open(USER['photo_name'], 'wb') as new_file:
            new_file.write(downloaded_file)
        image = cv2.imread(USER['photo_name'])
        scaned = vignette_filter_photo(image)
        cv2.imwrite(USER['photo_name'], scaned)
        photo = open(USER['photo_name'], 'rb')
        msg = bot.send_photo(USER['chat_id'], photo)
        #os.remove(USER['photo_name'])
        for i in USERS:
            if i['chat_id'] == chat_id:
                USERS.remove(i)

        markup = types.ReplyKeyboardMarkup()

        markup.add(types.KeyboardButton(DOCUMENT_SCAN_BUTTON))
        markup.add(types.KeyboardButton(SKETCH_BUTTON))
        markup.add(types.KeyboardButton(CARTUNIZE_BUTTON))
        markup.add(types.KeyboardButton(VIGNETTE_FILTER_BUTTON))

        bot.send_message(chat_id=chat_id, text="CHOOSE OPERATION TO PERFORM", reply_markup=markup)
    else:
        msg = bot.send_message(chat_id=chat_id, text="Send Photo only")
        bot.register_next_step_handler(msg, vignette_filter)



# bot.polling()


@app.route('/' + API_TOKEN , methods=['POST'])
def getMessage():
    bot.process_new_updates([telebot.types.Update.de_json(request.stream.read().decode("utf-8"))])
    return "!", 200

#https://api.telegram.org/bot<868081058:AAFSj3Q2diNtIJnd0pt1xtC02HhhP06qxRs>/deleteWebhook?url=https://17f6244e.ngrok.io/
@app.route("/")
def webhook():
    bot.remove_webhook()
    bot.set_webhook(url='https://43b557e6.ngrok.io/' + API_TOKEN)
    return "!", 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get('PORT', 5000)))