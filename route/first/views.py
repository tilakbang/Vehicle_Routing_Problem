from django.http.response import HttpResponse
from django.shortcuts import render
import pandas as pd
from geopy.distance import geodesic
from math import ceil
import requests
from io import BytesIO, StringIO
from django.core.files.base import ContentFile
import csv
from haversine import haversine


# Calculate the distance between each pair of nodes
def calculate_distance(lat1, lon1, lat2, lon2):
    return haversine((lat1, lon1), (lat2, lon2))

# Create your views here.
params={}

def index(request):
    return render(request, 'index.html')

def get_csv_temp(start_lat,start_lon,uploaded_file):
    data = pd.read_excel(uploaded_file, "Sheet1")
    n = data.shape[0]
    response = StringIO()
    writer = csv.writer(response)
    writer.writerow(["node","distance to depot","demand"])
    writer.writerow([0,0,0])
    for i in range(n):
        writer.writerow([i+1,calculate_distance(data["Latitude"][i],data["Longitude"][i],start_lat,start_lon),data["load"][i]])

    csv_file_temp = ContentFile(response.getvalue().encode('utf-8'))
    return csv_file_temp

def get_csv_temp_1(start_lat,start_lon,uploaded_file):
    data = pd.read_excel(uploaded_file, "Sheet1")
    n = data.shape[0]
    response = StringIO()
    writer=csv.writer(response)
    m=[i for i in range(n+1)]
    # sample=','.join(m)
    writer.writerow(m)
    for i in range(n):
        l=[]
        l.append(i+1)
        for j in range(n):
            l.append(calculate_distance(data["Latitude"][i],data["Longitude"][i],data["Latitude"][j],data["Longitude"][j]))
        writer.writerow(l)

    csv_file_temp_1 = ContentFile(response.getvalue().encode('utf-8'))
    return csv_file_temp_1

def Clarke_Wright(capacity,lat,long,uploaded_file):

    temp_csv=get_csv_temp(lat,long,uploaded_file)
    # read node data in coordinate (x,y) format
    nodes = pd.read_csv(temp_csv, index_col = 'node')
    nodes.rename(columns={"distance to depot":'d0'}, inplace = True)
    node_number = len(nodes.index) - 1
    nodes.head()

    # read pairwise distance
    temp_1_csv=get_csv_temp_1(lat,long,uploaded_file)

    pw = pd.read_csv(temp_1_csv, index_col = [0])
    pw.index.rename('',inplace = True)
    # print(pw)

    # calculate savings for each link
    savings = dict()
    for r in pw.index:
        for c in pw.columns:
            if int(c) != int(r):            
                a = max(int(r), int(c))
                b = min(int(r), int(c))
                key = '(' + str(a) + ',' + str(b) + ')'
                savings[key] = nodes['d0'][int(r)] + nodes['d0'][int(c)] - pw[c][r]

    # put savings in a pandas dataframe, and sort by descending
    sv = pd.DataFrame.from_dict(savings, orient = 'index')
    sv.rename(columns = {0:'saving'}, inplace = True)
    sv.sort_values(by = ['saving'], ascending = False, inplace = True)
    sv.head()

    # convert link string to link list to handle saving's key, i.e. str(10, 6) to (10, 6)
    def get_node(link):
        link = link[1:]
        link = link[:-1]
        nodes = link.split(',')
        return [int(nodes[0]), int(nodes[1])]

    # determine if a node is interior to a route
    def interior(node, route):
        try:
            i = route.index(node)
            # adjacent to depot, not interior
            if i == 0 or i == (len(route) - 1):
                label = False
            else:
                label = True
        except:
            label = False
        
        return label

    # merge two routes with a connection link
    def merge(route0, route1, link):
        if route0.index(link[0]) != (len(route0) - 1):
            route0.reverse()
        
        if route1.index(link[1]) != 0:
            route1.reverse()
            
        return route0 + route1

    # sum up to obtain the total passengers belonging to a route
    def sum_cap(route):
        sum_cap = 0
        for node in route:
            sum_cap += nodes.demand[node]
        return sum_cap

    # determine 4 things:
    # 1. if the link in any route in routes -> determined by if count_in > 0
    # 2. if yes, which node is in the route -> returned to node_sel
    # 3. if yes, which route is the node belongs to -> returned to route id: i_route
    # 4. are both of the nodes in the same route? -> overlap = 1, yes; otherwise, no
    def which_route(link, routes):
        # assume nodes are not in any route
        node_sel = list()
        i_route = [-1, -1]
        count_in = 0
        
        for route in routes:
            for node in link:
                try:
                    route.index(node)
                    i_route[count_in] = routes.index(route)
                    node_sel.append(node)
                    count_in += 1
                except:
                    pass
                    
        if i_route[0] == i_route[1]:
            overlap = 1
        else:
            overlap = 0
            
        return node_sel, count_in, i_route, overlap

    # create empty routes
    routes = list()

    # if there is any remaining customer to be served
    remaining = True

    # define capacity of the vehicle
    cap = capacity

    # record steps
    step = 0

    # get a list of nodes, excluding the depot
    node_list = list(nodes.index)
    node_list.remove(0)

    # run through each link in the saving list
    for link in sv.index:
        step += 1
        if remaining:

            # print('step ', step, ':')

            link = get_node(link)
            node_sel, num_in, i_route, overlap = which_route(link, routes)

            # condition a. Either, neither i nor j have already been assigned to a route, 
            # ...in which case a new route is initiated including both i and j.
            if num_in == 0:
                if sum_cap(link) <= cap:
                    routes.append(link)
                    node_list.remove(link[0])
                    node_list.remove(link[1])
                    # print('\t','Link ', link, ' fulfills criteria a), so it is created as a new route')
                # else:
                    # print('\t','Though Link ', link, ' fulfills criteria a), it exceeds maximum load, so skip this link.')

            # condition b. Or, exactly one of the two nodes (i or j) has already been included 
            # ...in an existing route and that point is not interior to that route 
            # ...(a point is interior to a route if it is not adjacent to the depot D in the order of traversal of nodes), 
            # ...in which case the link (i, j) is added to that same route.    
            elif num_in == 1:
                n_sel = node_sel[0]
                i_rt = i_route[0]
                position = routes[i_rt].index(n_sel)
                link_temp = link.copy()
                link_temp.remove(n_sel)
                node = link_temp[0]

                cond1 = (not interior(n_sel, routes[i_rt]))
                cond2 = (sum_cap(routes[i_rt] + [node]) <= cap)

                if cond1:
                    if cond2:
                        # print('\t','Link ', link, ' fulfills criteria b), so a new node is added to route ', routes[i_rt], '.')
                        if position == 0:
                            routes[i_rt].insert(0, node)
                        else:
                            routes[i_rt].append(node)
                        node_list.remove(node)
                    else:
                        # print('\t','Though Link ', link, ' fulfills criteria b), it exceeds maximum load, so skip this link.')
                        continue
                else:
                    # print('\t','For Link ', link, ', node ', n_sel, ' is interior to route ', routes[i_rt], ', so skip this link')
                    continue

            # condition c. Or, both i and j have already been included in two different existing routes 
            # ...and neither point is interior to its route, in which case the two routes are merged.        
            else:
                if overlap == 0:
                    cond1 = (not interior(node_sel[0], routes[i_route[0]]))
                    cond2 = (not interior(node_sel[1], routes[i_route[1]]))
                    cond3 = (sum_cap(routes[i_route[0]] + routes[i_route[1]]) <= cap)

                    if cond1 and cond2:
                        if cond3:
                            route_temp = merge(routes[i_route[0]], routes[i_route[1]], node_sel)
                            temp1 = routes[i_route[0]]
                            temp2 = routes[i_route[1]]
                            routes.remove(temp1)
                            routes.remove(temp2)
                            routes.append(route_temp)
                            try:
                                node_list.remove(link[0])
                                node_list.remove(link[1])
                            except:
                                #print('\t', f"Node {link[0]} or {link[1]} has been removed in a previous step.")
                                pass
                            # print('\t','Link ', link, ' fulfills criteria c), so route ', temp1, ' and route ', temp2, ' are merged')
                        else:
                            # print('\t','Though Link ', link, ' fulfills criteria c), it exceeds maximum load, so skip this link.')
                            continue
                    else:
                        # print('\t','For link ', link, ', Two nodes are found in two different routes, but not all the nodes fulfill interior requirement, so skip this link')
                        continue
                else:
                    # print('\t','Link ', link, ' is already included in the routes')
                    continue

            # for route in routes: 
                # print('\t','route: ', route, ' with load ', sum_cap(route))
        else:
            # print('-------')
            # print('All nodes are included in the routes, algorithm closed')
            break

        remaining = bool(len(node_list) > 0)

    # check if any node is left, assign to a unique route
    for node_o in node_list:
        routes.append([node_o])

    # add depot to the routes
    for route in routes:
        route.insert(0,0)
        route.append(0)

    # print('------')
    # print('Routes found are: ')

    # print(routes)
    return routes

def excelinput(request):
    if request.method=="POST":
        capacity = int(request.POST['capacity'])
        lat = float(request.POST['lat'])
        long = float(request.POST['long'])
        option = int(request.POST['format'])
        uploaded_file=request.FILES['document']
        # print(option, type(option))
        # print(lat, "Hello")
        # print(uploaded_file)
        if option==1:
            main_1(capacity,lat,long,uploaded_file)
        elif option==2:
            main_2(capacity,lat,long,uploaded_file)
        return render(request, 'nextpage.html', params)
    else:
        return HttpResponse("invalid")  

def clustering(capicity, df):
    # df = pd.read_excel("D:\\SY CS\\EDI\\Book2.xlsx", "Sheet1")
    lat = list(df["Latitude"])
    long = list(df["Longitude"])
    load = list(df["load"])
    cord = {(lat[i], long[i]): load[i] for i in range(len(lat))}
    # capicity = int(input("Enter capicity of vechicle :"))
    del_cords=[]
    for i in cord:
        if cord[i]>capicity:
            # print("The load at the point "+str(i)+" is greater than capicity so this point will not be included in any of the clusters")
            del_cords.append(i)
    params['msg']=del_cords
    for i in del_cords:
        del cord[i]
            
    cluster = ceil(sum(load) / capicity)
    all_include=True
    clusters={}
    while(all_include):
        count = 0
        centroid = []
        for i in cord:
            centroid.append(i)
            count += 1
            if (count == cluster):
                break
        clusters={i:[] for i in centroid}
        for z in range(2):
            for i in cord:
                ref_dic = {}
                centroid = list(clusters.keys())
                for j in centroid:
                    ref_dic[float(geodesic(i, j).km)] = (i, j)
                l = list(sorted(ref_dic))
                ref_dic = {i: ref_dic[i] for i in l}
                l1 = ref_dic.values()
                for k in l1:
                    ref_list = [cord[i] for i in clusters[k[1]]]
                    total = sum(ref_list) + cord[i]
                    c = k[1]
                    li = clusters[c]
                    del clusters[c]
                    newlat = (c[0] + i[0]) / 2
                    newlong = (c[1] + i[1]) / 2
                    clusters[(newlat, newlong)] = li
                    clusters[(newlat, newlong)].append(i)
                    break
            new_centroid=[]
            for i in clusters:
                min=geodesic(i,clusters[i][0]).km
                n_c=clusters[i][0]
                for j in clusters[i]:
                    temp=geodesic(i,j).km
                    if temp>min:
                        min=temp
                        n_c=j
                new_centroid.append(n_c)
            clusters={i:[] for i in new_centroid}
        for i in cord:
            ref_dic = {}
            centroid = list(clusters.keys())
            for j in centroid:
                ref_dic[float(geodesic(i, j).km)] = (i, j)
            l = list(sorted(ref_dic))
            ref_dic = {i: ref_dic[i] for i in l}
            l1 = ref_dic.values()
            for k in l1:
                ref_list = [cord[i] for i in clusters[k[1]]]
                total = sum(ref_list) + cord[i]
                if (total <= capicity):
                    c = k[1]
                    li = clusters[c]
                    del clusters[c]
                    newlat = (c[0] + i[0]) / 2
                    newlong = (c[1] + i[1]) / 2
                    clusters[(newlat, newlong)] = li
                    clusters[(newlat, newlong)].append(i)
                    break
        total_points=0
        for i in clusters:
            total_points+=len(clusters[i])
        if len(cord)==total_points:
            all_include=False
        else:
            cluster+=1
            # print(cluster)
    return clusters


def two_opt_2(route):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue  # changes nothing, skip then
                new_route = route[:]
                new_route[i:j] = route[j - 1:i - 1:-1]  # this is the 2woptSwap
                if route_dist_2(new_route) < route_dist_2(best):  # what should cost be?
                    best = new_route
                    improved = True
        route = best
    return best

def secondmin(dic):
    dist=list(sorted(dic.values()))
    for i in dic:
        if dic[i]==dist[1]:
            return i


def second_min(dic,minimal):
    ref_set={j for i in minimal for j in i}
    
    # ref_set={i for i in ref_dic if ref_dic[i] in ref_set}
    dist_dic={k for i in ref_set for k in dic if k[0]==i}
    dist_dic={k:geodesic(k[0],k[1]).km for k in dist_dic}
    sorted_values = sorted(dist_dic.values())  # Sort the values
    sorted_dict = {}
    for i in sorted_values:
        for k in dist_dic.keys():
            if dist_dic[k] == i:
                sorted_dict[k] = dist_dic[k]
    for i in sorted_dict:
        if i[0]!=i[1] and i not in minimal and  i[1] not in ref_set:
            return i


def minimum_spanning_tree(cluster):
    # print(len(cluster))
    minimal_spanning=[]
    dist_dic={(cluster[0],cluster[i]):geodesic(cluster[0],cluster[i]).km for i in range(0,len(cluster))}
    minimal_spanning.append(secondmin(dist_dic))
    dist_dic={(i,j):geodesic(i,j).km for i in cluster for j in cluster}
    # print(minimal_spanning)
    for i in range(len(cluster)-2):
        minimal_spanning.append(second_min(dist_dic, minimal_spanning))
    return minimal_spanning


def just_route(minimum_spanning_tree,starting_point):
    # starting_point = (18.6025509, 74.0047135)
    l=[]
    for i in minimum_spanning_tree:
        if i[0] not in l:
            l.append(i[0])
    max_dist = geodesic(l[0], starting_point).km
    last_point=l[0]
    for i in l:
        temp = geodesic(starting_point, i).km
        if temp > max_dist:
            max_dist = temp
            last_point = i
    m = (starting_point[1] - last_point[1]) / (starting_point[0] - last_point[0])
    c = starting_point[1] - m * starting_point[0]
    l=[]
    for i in minimum_spanning_tree:
        for j in i:
            if j not in l:
                l.append(j)
    l.remove(starting_point)
    l1 = []
    l2 = []
    for i in l:
        value = i[1] - m * i[0] - c
        if value > 0:
            l1.append(i)
        else:
            l2.append(i)
    l1.insert(0, starting_point)
    l2.insert(0, starting_point)
    for i in range(len(l2)):
        l1.append(l2[-1])
        l2.remove(l2[-1])
    return l1


def route_dist_2(l):
    dist=0
    print(l)
    for i in range(len(l)-1):
        dist=dist+geodesic(l[i],l[i+1]).km
    return dist



def main_1(capacity, lat, long, uploaded_file):
    routes=Clarke_Wright(capacity, lat, long, uploaded_file)
    print(routes)
    df = pd.read_excel(uploaded_file, "Sheet1")
    indexinginfile={}
    n = df.shape[0]
    indexinginfile[0]=(lat,long)
    for i in range(n):
        indexinginfile[i+1]=(df["Latitude"][i],df["Longitude"][i])
    print("Indexing in file",indexinginfile)
    
    urllist=[]
    ind_route_distance=[]
    total=0
    froutes=[]
    for route in routes:
        froute=""

        for i in range(len(route)):
            froute+=str(route[i])+" ==> "
            route[i]=indexinginfile[route[i]]
        froutes.append(froute)
        url=printurl(route)
        urllist.append(url)
        print("route",route)
        ind_route_distance.append(route_dist_2(route))
        total+=route_dist_2(route)
    print("final")
    print(urllist)
    print(froutes)
    print(ind_route_distance)
    params['all']=[(froutes[i],urllist[i],ind_route_distance[i]) for i in range(len(urllist))]
    params['totaldistance']=str(total)
    print('Total route distance is :'+str(total))
    # return HttpResponse("Hello")

def main_2(capacity, lat, long, uploaded_file):
    starting_point=(lat, long)
    df = pd.read_excel(uploaded_file, "Sheet1")
    clusters=clustering(capacity, df)
    print(clusters)
    lat = list(df["Latitude"])
    long = list(df["Longitude"])
    print(lat)
    indexinexcel = {(lat[i], long[i]): (i+1) for i in range(len(lat))}
    indexinexcel[starting_point]=0
    urllist=[]
    total=0
    routes=[]
    ind_route_distance=[]
    for i in clusters:
        if len(clusters[i])!=1:
            cluster=clusters[i].copy()
            cluster.insert(0,starting_point)
            minimumspanningtree=minimum_spanning_tree(cluster)
            l=just_route(minimumspanningtree,starting_point)
            route=two_opt_2(l)
            print(route)
            # plotting(route,a)
            url=printurl(route)
            urllist.append(url)
            total+=route_dist_2(route)
            # print(route)
            # print(indexing(indexinexcel,route))
            routes.append(indexing(indexinexcel,route))
            # print(route_dist_2(route))
            ind_route_distance.append(route_dist_2(route))
        else:
            print(i)

    print(routes)
    params['all']=[(routes[i],urllist[i],ind_route_distance[i]) for i in range(len(urllist))]
    params['totaldistance']=str(total)
    print('Total route distance is :'+str(total))

def printurl(route):
    url1="https://dev.virtualearth.net/REST/v1/Imagery/Map/Road/Routes?"
    key = 'Al7q1kp7YP-UbS2gM6u7R1sXtXgnH0rWBc_s_r17qnSCndLXvHepYOouw6BmPqyO'
    for i in range(len(route)):
        url1+="wp."+str(i)+"="+str(route[i][0])+','+str(route[i][1])+';66;'+str(i+1)+'&'
    url=url1+'key='+key
    return url

def indexing(indexinexcel,route):
    l=""
    for i in route:
        l+=str(indexinexcel[i])+" => "
    l=l[:-3]
    return l
