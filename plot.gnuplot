
set key autotitle columnhead
set term wxt 0
plot for [col=2:2] 'episodeLength' using 0:col with points
set term wxt 1
plot for [col=2:6] 'stateValues' using 0:col with lines
pause mouse close