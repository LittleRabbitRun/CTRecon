function out = isfm()
% freemat doesn't have the unix() function
out = ~exist( 'unix', 'builtin' );