data = randi([-15 10],1,100);
z_score = zscore(data);
figure
plot(data)
figure
plot(z_score);

max_z_score = max(z_score);
min_z_score = min(z_score);
norm = normalize(data, 'range', [min_z_score max_z_score]);
