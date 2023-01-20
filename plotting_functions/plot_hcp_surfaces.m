function plot_hcp_surfaces(atlas,sourcemodel,color,neg,values,viewL,viewR,title, color_limits)

sgtitle(title);

ax(1)=subplot(1,2,1);
cmap=[];
tmp = brewermap(361,color);

cmap = [0.9686 0.9843 1; tmp];
if neg==1
colormap(ax(1),flip(cmap))
else
colormap(ax(1),cmap)
end
%caxis([prctile(atlas.data,1) prctile(atlas.data,95)]);
%caxis([0 4]);
patch('Vertices', sourcemodel.pos, 'Faces', sourcemodel.tri,...
'FaceVertexCData',atlas.data','FaceColor','flat',...
'EdgeAlpha',0,'LineStyle','none',...
'EdgeColor','interp');
axis off
axis vis3d
axis equal 
view(viewL)
camlight
lighting gouraud
material dull
axis tight
clim(color_limits)


ax(2)=subplot(1,2,2);
if neg==1
colormap(ax(2),flip(cmap))
else
colormap(ax(2),cmap)
end
%caxis([prctile(atlas.data,1) prctile(atlas.data,95)]);
%caxis([0 4]);

patch('Vertices', sourcemodel.pos, 'Faces', sourcemodel.tri,...
'FaceVertexCData',atlas.data','FaceColor','flat',...
'EdgeAlpha',0,'LineStyle','none',...
'EdgeColor','interp');
axis off
axis vis3d
axis equal 
view(viewR)
camlight
lighting gouraud
material dull
axis tight
clim(color_limits)


cb=colorbar('SouthOutside');
pos = get(cb,'Position');
set(cb,'Position',[pos(1)/1.6 pos(2) pos(3) pos(4)])
cb.FontSize = 16;
cb.Title.String=values;
%cb.Ticks=round(cb.Limits);

end
