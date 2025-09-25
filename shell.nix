{ pkgs ? import <nixpkgs> {} }:

let
  # Custom OpenCV with GUI support - check available parameters first
  opencv-gui = pkgs.python312Packages.opencv4.override {
    enableGtk3 = true;
    enableGStreamer = true;
    enableFfmpeg = true;
  };
  
  # Python environment with custom packages
  pythonEnv = pkgs.python312.withPackages (ps: with ps; [
    numpy
    opencv-gui
    pynput
    pip
  ]);
in

pkgs.mkShell {
  buildInputs = with pkgs; [
    pythonEnv
    
    # System dependencies for OpenCV GUI
    xorg.libX11
    xorg.libXtst
    gtk3
    pkg-config
    libGL
    glib
    
    # Additional GUI dependencies
    qt5.qtbase
    xorg.libXext
    xorg.libXrender
    fontconfig
    freetype
    
    # Camera access
    v4l-utils
    ffmpeg
    
    # Development tools
    which
  ];

  shellHook = ''
    echo "OpenCV Computer Vision Pen Environment"
    echo "Python version: $(python --version)"
    
    # Set up environment variables for GUI applications
    export QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins"
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ 
      pkgs.xorg.libX11 
      pkgs.xorg.libXtst 
      pkgs.gtk3 
      pkgs.libGL 
      pkgs.xorg.libXext
      pkgs.xorg.libXrender
      pkgs.qt5.qtbase
    ]}:$LD_LIBRARY_PATH"
    
    # GUI environment
    export QT_QPA_PLATFORM="xcb"
    export GDK_BACKEND="x11"
    
    # Camera permissions
    export OPENCV_LOG_LEVEL=ERROR
    
    # Test OpenCV build info
    echo "Testing OpenCV GUI support..."
    python -c "import cv2; print('OpenCV version:', cv2.__version__); print('GUI backends:', cv2.getBuildInformation())" 2>/dev/null || echo "OpenCV check failed"
    
    echo "Environment ready! Use 'python main.py' to start."
  '';
}