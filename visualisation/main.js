function show_tab(tab_name) {
       if (tab_name != 'clusters')
         set_pause_state(true)

       d3.selectAll('.tab').style('z-index', '0')
       d3.select('#tab-' + tab_name).style('z-index', '1').classed('section-tabs-button-active', true)

       d3.selectAll('.section-tabs-button').classed('section-tabs-button-active', false)
       d3.select('#section-tabs-button-' + tab_name).classed('section-tabs-button-active', true)

     }
     

     // map initialization
     var map = L.map('mapid').setView([50.0, 30.0], 5);

     L.tileLayer('https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token=pk.eyJ1IjoibWFwYm94IiwiYSI6ImNpejY4NXVycTA2emYycXBndHRqcmZ3N3gifQ.rJcFIG214AriISLbB6B5aw', {
       maxZoom: 18,
       attribution: 'Map data &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors, ' +
                    '<a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, ' +
                    'Imagery © <a href="http://mapbox.com">Mapbox</a>',
       id: 'mapbox.streets'
     }).addTo(map);

     var popup = L.popup();

     var markers = L.markerClusterGroup();
     markers.maxClusterRadius = 10
     d3.json("coordinates_17.json").then(function(data) {
       for (let i = 0; i < data.l.length; i++) {
         point = data.l[i]//Math.floor(Math.random() * data.l.length)]
         marker = L.marker([point.coordinates[0], point.coordinates[1]])
         marker.bindPopup(point.text)
         markers.addLayer(marker)
         
       }
       map.addLayer(markers)
     })



     // cluster view initialization
     var ru_clusters = d3.map([], function(d) { return d.i })
     var fi_clusters = d3.map([], function(d) { return d.i })

     var ru_enabled = !!d3.select('#checkbox-ru-lang').property("checked")
     var fi_enabled = !!d3.select('#checkbox-fi-lang').property("checked")

     var cluster_snapshots = []
     var next_update_index = 0
     var cluster_snapshots_by_date = []
     var ru_cluster_snapshots = []
     var next_ru_update_index = 0
     var fi_cluster_snapshots = []
     var next_fi_update_index = 0
     var current_day = new Date('2014-07-02')

     // scale up Finnish clusters to compensate for ~6 times less messages
     var FI_SCALE_UP = Math.sqrt(6)


     var svg = d3.select('#cluster-svg')

     var width  = +svg.style('width').replace('px', '')
     var height = +svg.style('height').replace('px', '')

     var cluster_elements

     var zoom_factor = 1

     svg.call(d3.zoom()
                .extent([[0, 0], [width, height]])
                .translateExtent([[-width / 2, -height / 2], [width / 2, height / 2]])
                .scaleExtent([1, 16])
                .on("zoom", function() {
                  d3.select('#tooltip').style('visibility', 'hidden')
                  zoom_factor = d3.event.transform.k
                  update_global_transform()
                }))

     window.addEventListener("resize", function() {
       width  = +svg.style('width').replace('px', '')
       height = +svg.style('height').replace('px', '')
       create_grid_lines()
       //update_global_transform()
     })

     // create grid
     var box_size = 100
     var grid_group = svg.append('g')
     var h_grid_lines = grid_group.append('g')
     var v_grid_lines = grid_group.append('g')
     create_grid_lines()


     // close the tooltip if anything else is clicked
     svg.on('click', function() {
       d3.select('#tooltip').style('visibility', 'hidden')
     })

     var cluster_group = svg.append('g').attr('class', 'nodes')
     function update_global_transform() {
       cluster_group.attr("transform", d3.event.transform)

       // wrap the grid lines transform so that it looks endless
       grid_group.attr("transform", "translate(" + (d3.event.transform.x % (box_size * zoom_factor)) + "," + (d3.event.transform.y % (box_size * zoom_factor)) + ")scale(" + d3.event.transform.k + ")");
     }
     //update_global_transform()


     var sentiment_color = d3.scaleSequential(d3.interpolateRdYlBu).domain([-1.2, 1.2])

     // simulation setup with all forces
     var collisionForce = d3
       .forceCollide().radius(function(d) { return d.s; })

     var simulation = d3
       .forceSimulation()
       .force("x", d3.forceX(function(cluster) { return cluster.t_sne[0] * width / 2.5 /* cluster.sentiment_total * width / 3 */} ).strength(0.02))
       .force("y", d3.forceY(function(cluster) { return cluster.t_sne[1] * height / 2.5}).strength(0.02))
       .force("collide", collisionForce)
       .alphaTarget(0.2)


     function update_visible_clusters() {
       ru_enabled = !!d3.select('#checkbox-ru-lang').property("checked")
       fi_enabled = !!d3.select('#checkbox-fi-lang').property("checked")

       update_simulation()
     }

     function get_visible_clusters() {
       let visible = [];

       if (ru_enabled)
         visible = ru_clusters.values()

       if (fi_enabled) {
         if (visible.length == 0)
           visible = fi_clusters.values()
         else
           visible = visible.concat(fi_clusters.values())
       }

       return visible
     }

     function update_graph() {

       // remove finished ones with negative size
       let to_be_removed = []
       ru_clusters.each(function(cluster, id, map) { if (cluster.ts < 0 && cluster.s <= 0) to_be_removed.push(id) })
       for (let i = 0; i < to_be_removed.length; i++)
         ru_clusters.remove(to_be_removed[i])

       to_be_removed = []
       fi_clusters.each(function(cluster, id, map) { if (cluster.ts < 0 && cluster.s <= 0) to_be_removed.push(id) })
       for (let i = 0; i < to_be_removed.length; i++)
         fi_clusters.remove(to_be_removed[i])


       cluster_elements = cluster_group.selectAll('g.cluster')
                               .data(get_visible_clusters(), function (cluster) { return cluster.i })

       cluster_elements.exit().remove()

       let nodeEnter = cluster_elements
         .enter()
         .append('g')
         .classed('cluster', true)
         .on('click', show_cluster_tooltip)
         .attr("transform", 'scale(0)')

       nodeEnter
         .filter(function(cluster) { return cluster.lang == 'ru' })
         .append('circle')
         .attr('r', 10)
         .attr('fill', function(cluster) { return sentiment_color(cluster.sentiment) })
       
       nodeEnter
         .filter(function(cluster) { return cluster.lang == 'fi' })
         .append('rect')
         .attr('x', -9)
         .attr('y', -9)
         .attr('width', 18)
         .attr('height', 18)
         .attr('rx', 5)
         .attr('ry', 5)
         .attr('fill', function(cluster) { return sentiment_color(cluster.sentiment) })
       
       let keyword_group = nodeEnter.append('g').classed('cluster-gkeyword', true)
       for(let i = 0; i < 3; i++) {
         keyword_group
           .append('text')
           .text(function(cluster) {
             return cluster.k[i]
           })
           .attr('y', (i - 2) * 3 + 0.5)
           .classed('cluster-keyword', true)
       }
       for(let i = 0; i < 2; i++) {
         nodeEnter
           .append('text')
           .text(function(cluster) {
             if (cluster.tags)
               return cluster.tags[i]
             else
               return ''
           })
           .attr('y', i * 2.5 + 3.5)
           .classed('cluster-tag', true)
       }

       cluster_elements = nodeEnter.merge(cluster_elements)
     }

     var sentiment_mode = d3.select('input[name="sentiment_mode"]:checked').property('value')

     function sentiment_tick() {
       switch(sentiment_mode) {
         case 'total':
           cluster_elements.selectAll('circle').attr('fill', function(cluster) { return sentiment_color(cluster.sentiment_total) })
           cluster_elements.selectAll('rect').attr('fill', function(cluster) { return sentiment_color(cluster.sentiment_total) })
           break
         case 'accum':
           cluster_elements.selectAll('circle').attr('fill', function(cluster) { return sentiment_color(cluster.sentiment_accum) })
           cluster_elements.selectAll('rect').attr('fill', function(cluster) { return sentiment_color(cluster.sentiment_accum) })
           break
         case 'cont':
           cluster_elements.selectAll('circle').attr('fill', function(cluster) { return sentiment_color(cluster.sentiment) })
           cluster_elements.selectAll('rect').attr('fill', function(cluster) { return sentiment_color(cluster.sentiment) })
           break
       }
     }

     function update_sentiment_mode(mode) {
       sentiment_mode = mode
       sentiment_tick()
     }

     function update_simulation() {
       update_graph()

       simulation.nodes(get_visible_clusters()).on('tick', () => {
         cluster_elements.attr("transform", function(d) { 
           return 'translate(' + [d.x, d.y] + ') scale(' + d.s * 0.1 + ')'; 
         })
         sentiment_tick()
         collisionForce.radius( function(node) { return node.s; })
       })

       simulation.alpha(1).restart()
     }


     var paused = false
     d3.select('#button-pause').on('click', function() { set_pause_state(!paused) })

     function set_pause_state(pause_state) {
       paused = pause_state

       let pause_btn = d3.select('#button-pause')
       if (!paused) {
         pause_btn.text('Pause')
         window.requestAnimationFrame(step);
       } else {
         pause_btn.text('Play')
       }

       simulation.alphaTarget(paused ? 0 : 0.2).restart()
     }

     var last_step = null;
     var time = current_day.getTime()
     var time_factor = 1000
     var cluster_scale = 10
     var speed_up = 1
     var active_day = new Date('2014-07-17');
     
     
     function process_cluster_data(ncluster)
       {
           if ('n' in ncluster) {
             let new_clusters = ncluster.n

             for (let n in new_clusters) {
               let new_data = new_clusters[n]

               // put the longest keywords in the middle so that they fit better inside the circles
               if (new_data.k)
                    new_data.k.sort(function(a, b) { return a.length - b.length })
               else
                   new_data.k = ['']

               cluster = {
                 i: parseInt(n),
                 s: 0,
                 ts: Math.sqrt(new_data.s) * cluster_scale,
                 x: 0,
                 y: 0,
                 lang: new_data.lang,
                 k: new_data.k,
                 sentiment: new_data.sentiment,
                 sentiment_total: new_data.sentiment_total,
                 sentiment_accum: new_data.sentiment,
                 tags: new_data.tags,
                 t_sne: new_data.t_sne
                 
               }

               if (cluster.lang == 'ru')
                 ru_clusters.set(n, cluster)
               else if (cluster.lang == 'fi')
                 fi_clusters.set(n, cluster)
             }
             
           }

           if ('u' in ncluster) {
             let update = ncluster.u
             if (typeof(cluster_elements) != 'undefined')
                cluster_elements.filter(function(cluster) { return cluster.i in update })
                                                              .transition().duration(1000)
                                                              .tween('size', function(cluster) {
                                                                return cluster_param_interpolator(cluster, update)
                                                              })
                                                              .selectAll('g.cluster-gkeyword')
                                                              .selectAll('text')
                                                              .text(function(cluster, text_index) { return cluster.k[text_index] })
           }
       }
     
     function step(timestamp) {

       if (paused)
         return

       if (!last_step) last_step = timestamp
       let dt = timestamp - last_step 

       // cap delta time to 200ms
       if (dt > 200)
         dt = 200

       last_step = timestamp

       // TODO cluster_elements.filter does not work properly if using while?
       // cluster_elements.filter is returning elements that do not satisfy the condition.
       // Right now only one update per frame is processed which is not quick enough sometimes.
       if (next_update_index < cluster_snapshots.length) {
         if (cluster_snapshots[next_update_index].t >= time * 0.001) { // TODO Remove when while is implemented properly 
           /*if (visible_clusters.empty())
              speed_up = speed_up * 0.999 + 0.001 * 4
              else
              speed_up = speed_up * 0.99 + 0.01*/

           time += dt * time_factor * speed_up
         }

         d3.select('#time_text').text(new Date(time).toLocaleString('en-GB'))

         if (cluster_snapshots[next_update_index].t < time * 0.001) {
           
           
           if (cluster_snapshots[next_update_index].t < ((time - 200) * 0.001)){
               var delayed_clusters = [];
               
               while ((cluster_snapshots[next_update_index].t < ((time - 200) * 0.001)) && (next_update_index < (cluster_snapshots.length-1))) {
                   if ('n' in cluster_snapshots[next_update_index]){
                         let new_clusters = cluster_snapshots[next_update_index].n

                         for (let n in new_clusters) {
                             delayed_clusters[n] = new_clusters[n]
                         }
                     }
                     
                   if ('u' in cluster_snapshots[next_update_index]){
                         let update = cluster_snapshots[next_update_index].u

                         for (let n in update) {
                             if (update[n].s > 0)
                             {
                                 if (typeof(delayed_clusters[n]) != 'undefined')
                                 {
                                     delayed_clusters[n].s = update[n].s;
                                     delayed_clusters[n].sentiment_total = update[n].sentiment_total
                                     delayed_clusters[n].sentiment = update[n].sentiment
                                     if (typeof(update[n].k) != 'undefined')
                                        delayed_clusters[n].k = update[n].k;
                                     if (typeof(update[n].tags) != 'undefined')
                                        delayed_clusters[n].tags = update[n].tags;
                                 }
                             }
                             else
                             {
                                 delayed_clusters.pop(n);
                             }
                         
                         }
                     }
               next_update_index++;       
               }
               process_cluster_data({n: delayed_clusters});  
           }
           process_cluster_data(cluster_snapshots[next_update_index]);
           
           next_update_index++
         }
         update_simulation();
         window.requestAnimationFrame(step);
       }
       else
       {
            // Handle end of the day
            current_day.setDate(current_day.getDate()+1)
            getAndRun(current_day)
       }
     }
    
     function prepare_cluster_data(day){
         return d3.json('cluster_data_test.json?day='+day).then(function(data) {
           var cluster_snapshots = data

           
           
           cluster_snapshots_by_date[day] = cluster_snapshots
           return day
         })
     }
     
     
     function getAndRun(date)
     {
         prepare_cluster_data(date.toISOString()).then(function(day){
             if (day == date.toISOString()) 
                {
                    cluster_snapshots = cluster_snapshots_by_date[day];
                    active_day = date.toISOString();
                    ru_clusters = d3.map([], function(d) { return d.i })
                    fi_clusters = d3.map([], function(d) { return d.i })
                    // start at the first snapshot greater than current time
                    var i = 0;
                    while (time > cluster_snapshots[i].t * 1000)
                    {
                        i++;
                        if (typeof(cluster_snapshots[i]) == 'undefined'){
                            i = i-1;
                            break
                        }
                    }
                    next_update_index = 0
                    time = cluster_snapshots[i].t * 1000
                    window.requestAnimationFrame(step);
                }
         });
     }
     
     getAndRun(current_day);
 
     function show_cluster_tooltip(cluster) {
       d3.select('#tooltip-inner-table').html('')
       d3.select('#tooltip-info')
              .html('<span>Cluster: ' + cluster.i + ', fetching data...</span>')

       let outer_tooltip = d3.select('#tooltip')
                             .style('visibility', 'visible')
                             .style('top', d3.event.pageY+'px')

       let tp_width  = outer_tooltip.node().offsetWidth
       let tp_height = outer_tooltip.node().offsetHeight

       outer_tooltip.style('left', (Math.min(d3.event.pageX, window.innerWidth - tp_width) - 300/* tab margin */)+'px')
       outer_tooltip.style('top', Math.min(d3.event.pageY, window.innerHeight - tp_height)+'px')

       d3.json('cluster_data/cluster_' + cluster.i + '.json?day='+active_day+'&time='+(time+10800*1000)).then(function(data) {

         cluster_info_str = '<span>Cluster: ' + cluster.i + ', documents: ' + data.length + '<br>Keywords:'

         for (let i = 0; i < cluster.k.length; i++)
           cluster_info_str += ' ' + cluster.k[i]

         cluster_info_str += '<br>Tags:'
         if (cluster.tags != null) {
           for (let i = 0; i < cluster.tags.length; i++)
             cluster_info_str += ' ' + cluster.tags[i]
         }

         cluster_info_str += '</span>'

         d3.select('#tooltip-info').html(cluster_info_str)

         let table = d3.select('#tooltip-inner-table')
                       .selectAll('tr')
                       .data(data)
                       .enter()
                       .append('tr')
         table.append('td')
                            .html(function(d) { return '<a href="https://twitter.com/' + d.screen_name + '" target="_blank">' + d.screen_name + '</a>'})
                            .classed('td-tweet-user', true)
         table .append('td')
                            .html(function(d) { return linkifyHtml(d.text, linkify.options.default) })
       }, function(reason) {
         d3.select('#tooltip-info')
           .html('<span>Cluster: ' + cluster.i + ', failed to fetch data: ' + reason.message + '.</span>')
       })

       d3.event.stopPropagation()
     }

     function create_grid_lines() {

       // vertical
       let num_lines = width / box_size
       let grid_arr = d3.range(0, num_lines + 1)
       v_grid_lines.selectAll("line").remove()
       let grid_selection = v_grid_lines.selectAll("line").data(grid_arr)

       grid_selection.enter().append("line").attr("x1", function (d){ return d * box_size}).attr("x2", function (d){return d * box_size }).attr("y1", -box_size).attr("y2", height + box_size).style("stroke", "#eee");


      // horizontal
      num_lines = height / box_size
      grid_arr = d3.range(0, num_lines + 1)
      h_grid_lines.selectAll("line").remove()
      grid_selection = h_grid_lines.selectAll("line").data(grid_arr)

      grid_selection.enter().append("line").attr("x1", -box_size).attr("x2", width + box_size).attr("y1", function (d){ return d * box_size}).attr("y2", function (d){ return d * box_size }).style("stroke", "#eee");
       
     }

     function cluster_param_interpolator(cluster, cluster_update) {
       let new_sentiment = cluster.sentiment
       let new_sentiment_accum = cluster.sentiment_accum

       if (cluster_update[cluster.i].s >= 0) {
         cluster.ts = Math.sqrt(cluster_update[cluster.i].s) * cluster_scale

         if (cluster.lang == 'fi')
           cluster.ts *= FI_SCALE_UP

         new_sentiment = cluster_update[cluster.i].sentiment
         new_sentiment_accum = cluster_update[cluster.i].sentiment_accum
       } else {
         // sentiment is not updated on the last update
         cluster.ts = cluster_update[cluster.i].s
       }

       if (cluster_update[cluster.i].k != null)
         if (cluster_update[cluster.i].k)
            cluster.k = cluster_update[cluster.i].k.sort(function(a, b) { return a.length - b.length })

       let size_i = d3.interpolate(cluster.s, Math.max(cluster.ts, 0))
       let sentiment_i = d3.interpolate(cluster.sentiment, new_sentiment)
       let sentiment_accum_i = d3.interpolate(cluster.sentiment_accum, new_sentiment_accum)
       return function(t) {
         cluster.s = size_i(t)
         cluster.sentiment = sentiment_i(t)
         cluster.sentiment_accum = sentiment_accum_i(t)
       }
     }
