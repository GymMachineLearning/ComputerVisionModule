## MOVNET + HOURGLASS

### current pipe: video -> movnet2d -> hourglass 3d

### TO DO:
#### add bar tracking in 3d
#### fix posture problems when some of the joints are hidden behind bar
#### model estimates posture using flip operations (dive into code)
#### add some assumptions (person want move, the whole movement on gym exercises is quite static, hands are connected with bar for the whole movement) if this can improve quality of detection