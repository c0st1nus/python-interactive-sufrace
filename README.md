# OpenCV Computer Vision Pen

Проект для управления курсором мыши с помощью цветового трекинга через веб-камеру.

## Установка на NixOS

### Вариант 1: Использование nix-shell (рекомендуется)

```bash
# Войти в среду разработки (автоматически удалит pip opencv-python)
nix-shell

# Установить только дополнительные Python пакеты
pip install -r requirements.txt

# Запуск (используется системный OpenCV с GUI поддержкой)
python main.py
```

### Вариант 2: Системная установка

Добавьте в `/etc/nixos/configuration.nix`:

```nix
environment.systemPackages = with pkgs; [
  python312
  python312Packages.opencv4
  python312Packages.numpy
  python312Packages.pynput
  
  # Системные зависимости
  xorg.libX11
  xorg.libXtst
  gtk3
];
```

Затем:
```bash
sudo nixos-rebuild switch
python main.py
```

## Требования

- NixOS с X11 (Wayland может не работать с pynput)
- Веб-камера
- Python 3.12+

## Использование

1. Запустите программу: `python main.py`
2. Покажите камере 4 опорных точки красного цвета по углам экрана
3. После калибровки красный объект будет управлять курсором мыши

## Устранение неполадок

- Если мышь не работает, проверьте права доступа к X11
- Для Wayland может потребоваться дополнительная настройка
- Убедитесь, что камера не используется другими приложениями

## Изменения для Linux

- Заменен `winsound` на системные звуки Linux  
- Заменен `win32api` на `pynput` для управления мышью
- Заменен `GetSystemMetrics` на `xrandr`/`xdpyinfo`
- Исправлен устаревший `time.clock()` на `time.perf_counter()`