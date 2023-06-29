import graphviz


ps = graphviz.Digraph('hueta', node_attr={'shape': 'plaintext', 'color': '#2d137d', 'fontcolor': '#2d137d',
                                          'fontsize':'14.0', 'size':'2', 'fontname':'monospace', 'font-weight':"bold"},
                      edge_attr={'color':'#2d137d', 'fontcolor': '#2d137d', 'fontname':'Courier', 'fontsize':'12.0'})

ps.node(name='start', image='', label='')

ps.node(name='krug_blyat', label=r'')

ps.node('A_SUBMITTED', label=r"A_SUBMITTED\n126\l" )
ps.node('A_PARTLYSUBMITTED', label=r"A_PARTLYSUBMITTED\n126\l")
ps.node('W_Afhandelen leads', label=r"W_Afhandelen leads\n378\l")
ps.node('A_PREACCEPTED', label=r"A_PREACCEPTED\n126\l")
ps.node('W_Completeren aanvraag', label=r"W_Completeren aanvraag\n630\l")
ps.node(name='end', image='', label='')

ps.edge('start', 'A_SUBMITTED', style="dashed")
ps.edge('A_SUBMITTED', 'A_PARTLYSUBMITTED', label="0.0d")
ps.edge('A_PARTLYSUBMITTED', 'W_Afhandelen leads', label="0.0d")
ps.edge('W_Afhandelen leads', 'W_Afhandelen leads', label="0.26d")
ps.edge('W_Afhandelen leads', 'A_PREACCEPTED', label="0.0d")
ps.edge('W_Afhandelen leads', 'W_Completeren aanvraag', label="0.18d")
ps.edge('A_PREACCEPTED', 'W_Afhandelen leads')
ps.edge('W_Completeren aanvraag', 'W_Afhandelen leads', label="0.0d")
ps.edge('A_PREACCEPTED', 'W_Completeren aanvraag', label="0.0d")
ps.edge('W_Completeren aanvraag', 'W_Completeren aanvraag', label="0.22d")
ps.edge('W_Completeren aanvraag', 'end', style="dashed")

ps.attr(rankdir='LR')

ps.format = 'svg'
ps.render()


