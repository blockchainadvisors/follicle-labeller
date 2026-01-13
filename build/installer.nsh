!macro customInstall
  ; Register .fol file association
  WriteRegStr SHCTX "Software\Classes\.fol" "" "FollicleLabeller.Project"
  WriteRegStr SHCTX "Software\Classes\FollicleLabeller.Project" "" "Follicle Labeller Project"
  WriteRegStr SHCTX "Software\Classes\FollicleLabeller.Project\DefaultIcon" "" "$INSTDIR\${APP_EXECUTABLE_FILENAME},0"
  WriteRegStr SHCTX "Software\Classes\FollicleLabeller.Project\shell\open\command" "" '"$INSTDIR\${APP_EXECUTABLE_FILENAME}" "%1"'

  ; Refresh shell icons
  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, i 0, i 0)'
!macroend

!macro customUnInstall
  ; Remove .fol file association
  DeleteRegKey SHCTX "Software\Classes\.fol"
  DeleteRegKey SHCTX "Software\Classes\FollicleLabeller.Project"

  ; Refresh shell icons
  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, i 0, i 0)'
!macroend
