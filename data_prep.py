import librosa      # для работы с аудио-файлами
import numpy as np  # для работы с аудио-файлами
import os           # для работы с файлами
import pathlib      # для работы с директорикй
max = 0.00          # максимальная длительность аудио, для Tacotron
transcript = open("/content/drive/MyDrive/ISSAI_KazakhTTS/transcript.txt", 'w')           # задаем файл для сбора текстов
for path in pathlib.Path("/content/drive/MyDrive/ISSAI_KazakhTTS/Transcripts").iterdir(): # пробегаемся по всем текстовым файлам
  if (path.is_file()):
    wav = os.path.splitext(path)[0]                         # извлекаем название текущего файла
    wav = wav.replace("Transcripts", "Audios")              # меняем директорию
    wav = wav + ".wav"                                      # конечный путь к соответствующему аудио
    f = open(path, 'r')
    s = f.read()                                            # считываем строку из текстового файла
    f.close()
    samples, rate = librosa.load(wav)                       # открываем аудио по ранее извлеченному пути
    time = np.arange(0, len(samples)) / rate                # "слушаем" аудио
    duration = time[-1]                                     # определяем длительность
    duration = int(duration * 100) / 100                    # приводим к удобной форме записи XX.XX
   if duration > max:
      max = duration                                        # фиксируем максимальную длительность
   p = wav + '|' + s + '|' + s + '|' + str(duration) + '\n' # формируем строку из пути к аудио, содержимого текущего файла и длительности
   transcript.write(p)                                      # записываем ее в файл
transcript.close()
print(f"Done, maxlen = {max}\n")
