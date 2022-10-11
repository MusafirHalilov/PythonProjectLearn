from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor

import os

bot = Bot(token=os.getenv('TOKEN'))
dp = Dispatcher(bot)


async def on_startup():
    print('Бот вышел в онлайн')



@dp.message_handler(commands=['start', 'help'])
async def command_start(message: types.Message):
    try:
        await bot.send_message(message.from_user.id, 'Приятного аппетита!')
        await message.delete()
    except:
        await message.reply('Общение с ботом через ЛС, напишите ему:\nhttps://t.me/P_S_MBot')


@dp.message_handler(commands=['Режим работы'])
async def pizza_open_command(message: types.Message):
    await bot.send_message(message.from_user.id, 'Вс-Чт с 9:00 до 20:00, Пт-Сб с 10:00 до 23:00')


@dp.message_handler(commands=['Расположение'])
async def pizz_place_comand(message: types.Message):
    await bot.send_message(message.from_user.id, 'ул. Колбасная 15')



@dp.message_handler()
async def echo_send(message: types.Message):
    if message.text == 'Привет':
        await message.answer('И тебе привет!')


#    await message.answer(message.text)
#    await message.reply(message.text)
#    await bot.send_message(message.from_user.id, message.text)

executor.start_polling(dp, skip_updates=True, on_startup=on_startup)