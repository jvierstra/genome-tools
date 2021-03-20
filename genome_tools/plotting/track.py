# Copyright 2016 Jeff Vierstra

import sys

from itertools import cycle

import numpy as np

import matplotlib
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib.lines as lines
import matplotlib.collections as collections

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def get_color_cycle():
    """Return the list of colors in the current matplotlib color cycle."""
    cyl = matplotlib.rcParams['axes.prop_cycle']
    return [x['color'] for x in cyl]


class _track(object):
    """Base level track object"""
    def __init__(self, 
                interval, 
                coord_locator=None, 
                value_locator=None,
                scale_bar_props=None,
                **kwargs):

        self.interval = interval

        #
        if coord_locator is None:
            self.coord_locator = ticker.MaxNLocator(3)
        elif isinstance(coord_locator, ticker.Locator):
            self.coord_locator = coord_locator
        else:
            raise TypeError("coord_locator must be an instance of matplotlib.ticker.Locator") 

        #
        if value_locator is None:
            self.value_locator = ticker.MaxNLocator(3)
        elif isinstance(value_locator, value.Locator):
            self.value_locator = value_locator
        else:
            raise TypeError("value_locator must be an instance of matplotlib.ticker.Locator") 

        #self.scale_bar_props = {}
        #self.scale_bar_props.update(scale_bar_props)
        
        self.options = {
            'facecolor': 'lightgrey',
            'edgecolor': 'grey',
            'linewidth': 1,
            'linestyle': '-',
        }

        self.options.update(kwargs)

    def format_axis(self, ax):

        # set defaults
        ax.ticklabel_format(style='plain', axis = 'both', useOffset=False)
                
        # tick direction out
        ax.xaxis.set_tick_params(direction = 'out')
        ax.yaxis.set_tick_params(direction = 'out')
        
        # set x limits to interval
        ax.set_xlim(left=self.interval.start, right=self.interval.end)
        
        # x-axis choose ~3 tick positions
        ax.xaxis.set(major_locator = self.coord_locator)
        ax.xaxis.set(minor_locator = ticker.MaxNLocator(15))
        ax.yaxis.set(major_locator = self.value_locator)
        
        # hide both x and y axis
        ax.xaxis.set(visible = False)
        ax.yaxis.set(visible = False)


    def format_spines(self, ax, remove_spines):
        all_spines = ['top', 'bottom', 'left', 'right']
        
        for spine in all_spines:
            ax.spines[spine].set_linewidth(0.5)

        for spine in remove_spines:
            try:
                ax.spines[spine].set_visible(False)
            except KeyError:
                pass

    def render(self, ax):
        # Remove the white patch behind each axes
        ax.patch.set_facecolor('none')

        # Add scale bar -- code is placed here so a scale bar can be drawn on any track instance
        # if self.options['scale_bar'] is not None:

        rough_bar_size = len(self.interval) / 10

        scale = np.floor(np.log10(rough_bar_size))
        multiplier = round(len(self.interval)/(10**scale)/10)

        final_bar_size = multiplier*10**scale

        bar = AnchoredSizeBar(ax.transData,
            final_bar_size, 
            label="%d nt" % final_bar_size, 
            loc=3,
            frameon=False)
        ax.add_artist(bar)



class _continuous_data_track(_track):

    def __init__(self, 
                interval, 
                data=None,
                vmin=None,
                vmax=None, 
                **kwargs):

        super(_continuous_data_track, self).__init__(interval, **kwargs)
        self.data = data.get(self.interval)
        
        self.vmin = np.nanmin(self.data) if not vmin else vmin
        self.vmax = np.nanmax(self.data) if not vmax else vmax

    def format_axis(self, ax):
        super(_continuous_data_track, self).format_axis(ax)

        ax.yaxis.set(visible = True)

    def render(self, ax):

        if self.data is None:
            raise Exception("No data loaded!")

        self.format_axis(ax)
        self.format_spines(ax, remove_spines=['top', 'right'])

        # if 'density' in self.options:
        #   xx, yy = self.density(self.data, window_size = self.options['density']['window_size'])
        #   xs = self.step(xx, xaxis = True)
        #   ys = self.step(yy) / self.options['density']['norm_factor']
        # else:
        #   xs = self.step(np.arange(self.interval.start, self.interval.end), xaxis = True)
        #   ys = self.step(self.data)

        # (ybot, ytop) = ax.get_ylim()
        # ys[ys > ytop] = ytop
        # ys[ys < ybot] = ybot

        # if 'fill_between' in self.options:
        #   ax.fill_between(xs, np.ones(len(xs)) * self.options["fill_between"], ys, 
        #       facecolor=self.options["facecolor"], 
        #       edgecolor=self.options["edgecolor"],
        #       linewidth=self.options["lw"],
        #       linestyle=self.options["ls"])
        # else:
        #   ax.plot(xs, y, 
        #       color=self.options["edgecolor"],                
        #       lw=self.options["lw"],
        #       ls=self.options["ls"])

        y = self.data.copy()

        y[y<self.vmin] = self.vmin 
        y[y>self.vmax] = self.vmax
        
        x = np.linspace(self.interval.start, self.interval.end, len(self.data))

        ax.fill_between(x, 0, y, step="mid", **self.options)

        ax.set_ylim(bottom=self.vmin, top=self.vmax)


        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for a, b in segment(self.data, self.vmax):
            ax.plot([x[a], x[b]], [1, 1], color = "hotpink", lw = 3, transform = trans, clip_on = False)

        super(_continuous_data_track, self).render(ax)


class _pileup_track(_track):

    def __init__(self, interval, palette=["lightgrey", "red", "blue"], **kwargs):
        super(_pileup_track, self).__init__(interval, **kwargs)

        self.segments = []

        self.palette = palette
        if self.palette is None:
            self.palette = get_color_cycle()

    def format_axis(self, ax):
        super(_pileup_track, self).format_axis(ax)

        ax.xaxis.set(visible = True)
        ax.yaxis.set(visible = False)

    def render(self, ax, pack = True, labels = False):

        self.format_axis(ax)
        self.format_spines(ax, remove_spines=['top', 'right'])


        levels = np.unique([i.score for i in self.segments])
        n_colors = len(levels)

        pal_cycle = cycle(self.palette)
        palette = [next(pal_cycle) for _ in range(n_colors)]

        palette = map(colors.colorConverter.to_rgb, palette)
        palette = {l:p for l,p in zip(levels, palette)}
        
        nrows, rows = pack_rows(self.segments)
        
        n_inverse = np.arange(nrows+1)[::-1]

        for segment in self.segments:

            n = n_inverse[rows[segment]] if pack else 0
                            
            x0 = segment.start
            x1 = segment.end
            y0 = -0.2+n
            w = x1-x0
            h = 0.4

            label_x, label_y, label_rot = x1, y0, 0

            p = patches.Rectangle((x0, y0), w, h, 
                edgecolor="k", 
                facecolor=palette[segment.score],
                lw=0,
                ls="solid",
                zorder=1)

            ax.add_patch(p)

        ax.set_ylim(bottom=-1.2, top=(nrows if pack else 1)+1.2)
        #ax.invert_yaxis()

        super(_pileup_track, self).render(ax)

class _gene_annotation_track(_track):

    def __init__(self, interval, data=None, **kwargs):
        super(_gene_annotation_track, self).__init__(interval, **kwargs)

        self.genes = {}
        self.transcripts = {}
        self.exons = {}
        self.cds = {}
        self.utrs = {}

        if data:
            genes, transcripts, exons, cds, utrs = data.get(interval)

            self.genes = genes
            self.transcripts = transcripts
            self.exons = exons
            self.cds = cds
            self.utrs = utrs

    
    def get_transcript_intervals(self):
        transcript_intervals = []
        for gene, trans in self.transcripts.items():
            for transcript_name, transcript_interval in trans:
                transcript_intervals.append(transcript_interval)

        return transcript_intervals

    def format_axis(self, ax):

        super(_gene_annotation_track, self).format_axis(ax)

        ax.xaxis.set(visible = False)
        ax.yaxis.set(visible = False)
        
    def render(self, ax, padding = 5):

        self.format_axis(ax)
        self.format_spines(ax, remove_spines=['top', 'bottom', 'left', 'right'])
        # pack transcripts

        arrowprops = dict(arrowstyle="<|-", connectionstyle="angle,angleA=0,angleB=90,rad=2")

        trans = self.get_transcript_intervals()
        nrows, rows = pack_rows(trans, padding = padding)


        lines_transcripts = []
        patches_cds = []
        patches_utrs = []

        for gene_id, transcripts in self.transcripts.items():

            for transcript_id, transcript_interval in transcripts:

                n = rows[transcript_interval]

                x0 = transcript_interval.start #- self.interval.start
                x1 = transcript_interval.end  #- transcript_interval.start + x0
                y = n

                over_left = x0 < self.interval.start #0
                over_right = x1 > self.interval.end #len(self.interval)
                start_visible = False

                x0 = max(x0, self.interval.start)
                x1 = min(x1, self.interval.end)
            
                if transcript_interval.strand == "-":
                    x0, x1 = x1, x0
                    direction=-1
                    ha = "right"
                else:
                    direction=1
                    ha = "left"


                if over_left:
                    ax.plot(x0, y, '>' if direction>0 else '<')
                if over_right:
                    ax.plot(x1, y, '>' if direction>0 else '<', clip_on=False, color='k')

                gene_name = self.genes.get(gene_id, transcript_id)

                lines_transcripts.append([(x0, y), (x1, y)])

                ax.annotate(gene_name, xy=(x0, y), xytext=(direction*15, 10), 
                    textcoords="offset points", ha=ha, arrowprops=arrowprops, 
                    fontstyle="italic", zorder=20)

                 # Loop through exons
                for (exon, exon_interval) in self.exons.get(transcript_id, []):

                    for (utr, utr_interval) in self.utrs.get(exon, []):
                        p = patches.Rectangle((utr_interval.start, -0.2+n), utr_interval.end-utr_interval.start, 0.4)
                        patches_utrs.append(p)

                    for (cds, cds_interval) in self.cds.get(exon, []):
                        p = patches.Rectangle((cds_interval.start, -0.3+n), cds_interval.end-cds_interval.start, 0.6)
                        patches_cds.append(p)

        ax.add_collection(collections.LineCollection(lines_transcripts, colors='k', linestyles='solid', linewidths=2))
        ax.add_collection(collections.PatchCollection(patches_cds, facecolors="gold", edgecolors="none", zorder=10))
        ax.add_collection(collections.PatchCollection(patches_utrs, facecolors="gold", edgecolors="none", zorder=10))

        ax.set_ylim(bottom = -0.5, top = nrows + 1.5)
        
        super(_gene_annotation_track, self).render(ax)
        


class _annotation(object):
    def __init__(self, x, text):
        self.pos = x
        self.text = text

    def render(self, ax):

        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

        ax.annotate(self.text, xy=(self.pos, 0),  xycoords='data',
            xytext=(-50, 50), textcoords="offset points",
            arrowprops=dict(facecolor='black', arrowstyle="-|>", connectionstyle="angle,angleA=0,angleB=90,rad=2"),
            bbox = dict(boxstyle="round", fc="0.8"),
            horizontalalignment='center', verticalalignment='top',
            )

class row_element(object):

    def __init__(self, interval, prev = None, next = None, padding = 5):
        self.interval = interval
        self.start = interval.start - padding
        self.end = interval.end + padding
        self.prev = prev
        self.next = next
        
class row(object):
    """ """
    def __init__(self, num):
        self.num = num
        self.elements = []
        self.first = None
        self.last = None
        
    def add_element(self, elem):
        
        if self.first is None:
            #sys.stderr.write("added feat %d-%d as only element of row %d\n" %
            #                (elem.start, elem.end, self.num))
            elem.prev = None
            elem.next = None
            self.first = self.last = elem
            return True
        
        cur = self.first
        while (cur and cur.start < elem.start):
            cur = cur.next
            
        if cur is None:
            if self.last.end < elem.start:
                #sys.stderr.write("added feat %d-%d to end of row %d\n" %
                #                 (elem.start, elem.end, self.num))
                elem.prev = self.last
                elem.next = None
                self.last.next = elem
                self.last = elem
                return True
            else:
                return False
        
        prev = cur.prev
        if prev is None:
            if elem.end < cur.start:
                #sys.stderr.write("added feat %d-%d to start of row %d "
                #                 "(before %d-%d)\n" %
                #                 (elem.start, elem.end, self.num,
                #                 cur.start, cur.end))
                elem.prev = None
                elem.next = cur
                cur.prev = elem
                self.first = elem
                return True
            else:
                return False
        
        if prev.end < elem.start and cur.start > elem.end:
            #sys.stderr.write("added feat %d-%d between %d-%d and %d-%d "
            #                 "in row %d\n" %
            #                 (elem.start, elem.end,
            #                  prev.start, prev.end,
            #                  cur.start, cur.end, self.num))
            elem.prev = prev
            elem.next = cur
            prev.next = elem
            cur.prev = elem
            return True
        
        return False

def pack_rows(intervals, padding = 5):
    
    rows = []
    row_num = {}
    
    nrows = -1

    for interval in intervals:
        #sys.stderr.write("adding feat...\n")
        re = row_element(interval, padding = padding)
        
        placed = False
        
        for r in rows:
            if(r.add_element(re)):
                placed = True
                row_num[interval] = r.num
                break
        
        if not placed:
            nrows = nrows + 1
            r = row(num = nrows)
            rows.append(r)
            row_num[interval] = r.num
            if not r.add_element(re):
                raise Exception("Could not place element in new row!")
                
    return nrows, row_num

def segment(x, threshold, w = 0, decreasing = False):
    """Segment an array into continuous elements passing a threshhold

    :returns: [(int, int), ...]: start and end points to regions that pass a threshold
    """
    dir = -1 if decreasing else 1

    ret = []
    curr_start = -1
    
    for i in range(x.shape[0]):
        if curr_start < 0:
            if dir*x[i] >= dir*threshold:
                curr_start = i-w+1
        else:
            if dir*x[i] < dir*threshold:
                if len(ret) > 0 and curr_start <= ret[-1][1]:
                    ret[-1][1] = i-1+w
                else:
                    ret.append( [curr_start, i-1+w] )
                curr_start = -1
    return ret
