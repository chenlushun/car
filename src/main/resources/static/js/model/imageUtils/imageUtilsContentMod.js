define(['api', 'utils'], function(api, utils){

    function init(){
        initTree();

        bindBtnEvent();
    }

    var treeId = "#treeId";
    var plateTreeNode = null;
    var dirTreeNode = null;
    var hsvValue = {};

    var c = document.getElementById("canvas");
    var ctxt = c.getContext('2d');

    function bindBtnEvent(){
        $("#canvas").on('click', function (evt) {
            var p = this;
            var _x = 0, _y = 0;
            while(p.offsetParent){
                _x += p.offsetLeft;
                _y += p.offsetTop;
                p = p.offsetParent;
            }
            _x += p.offsetLeft; // 图片起点坐标
            _y += p.offsetTop;  // 图片起点坐标

            $("#c_clos").val($(this).width());
            $("#c_rows").val($(this).height());
            $("#p_clos").val(evt.clientX - _x); // 鼠标点击位置相对起点坐标
            $("#p_rows").val(evt.clientY - _y); // 鼠标点击位置相对起点坐标

            var data = ctxt.getImageData(evt.clientX - _x, evt.clientY - _y, 1, 1).data;
            for(var i =0,len = data.length; i<len;i+=4){
                var red = data[i],//红色色深
                    green = data[i+1],//绿色色深
                    blue = data[i+2],//蓝色色深
                    alpha = data[i+3];//透明度
                console.log(red, green, blue, alpha);
                $("#rgbValue").val(red + ', ' + green + ', ' + blue + ', ' + alpha);
            }
            $("#rgbColor").style("backageground-color", "rgba("+ red + ', ' + green + ', ' + blue + ', ' + alpha + ")");
        });
    }

    function initTree() {
        isFirst = false; //加载树的时候默认咱开第一层级
        $.fn.zTree.destroy(treeId);
        $.fn.zTree.init($(treeId), setting);
    }
    
    function initSilder(data) {
        var hmin = ZUI.silder({
            elem: '.hmin',
            color: 'green',
            pos: data,
            showNum: true,
            count: 360,
            disable: false,
            callBackMove: function (num) {
                 // console.log('move', num);
            },
            callBackMouseup: function (num) {
                 // console.log('up', num);
                 // console.log('this', this);
                 // console.log('this', this.elem.attr("class"));
            }
        });
        ZUI.silder({
            elem: '.hmax',
            color: 'green',
            pos: '25%',
            showNum: true,
            count: 360,
            disable: false,
            callBackMouseup: function (num) {
                // console.log('up', num);
            }
        });
        ZUI.silder({
            elem: '.smin',
            color: 'red',
            pos: '25%',
            showNum: true,
            count: 255,
            disable: false,
            callBackMouseup: function (num) {
                // console.log('up', num);
            }
        });
        ZUI.silder({
            elem: '.smax',
            color: 'red',
            pos: '25%',
            showNum: true,
            count: 255,
            disable: false,
            callBackMouseup: function (num) {
                // console.log('up', num);
            }
        });
        ZUI.silder({
            elem: '.vmin',
            color: 'blue',
            pos: '25%',
            showNum: true,
            count: 255,
            disable: false,
            callBackMouseup: function (num) {
                // console.log('up', num);
            }
        });
        ZUI.silder({
            elem: '.vmax',
            color: 'blue',
            pos: '25%',
            showNum: true,
            count: 255,
            disable: false,
            callBackMouseup: function (num) {
                // console.log('up', num);
            }
        });
    }

    // 树结构配置
    var setting = {
        edit: {
            enable: true,
            editNameSelectAll: true,
            showRemoveBtn: true,
            showRenameBtn: true
        },
        view: {
            addHoverDom: addHoverDom,
            removeHoverDom: removeHoverDom
        },
        check: {
            enable: false
        },
        callback: {
            onClick: treeClick,
            onAsyncSuccess:onAsyncSuccess,
            beforeRemove: beforeRemove,
            beforeRename: beforeRename,
        },
        async: {
            enable: true,
            url: api.file.getFileTreeByDir,
            type: 'get',
            dataType: "json",
            autoParam: ["filePath=dir"],
            otherParam: {"typeFilter":"png,jpg,jpeg"},
            dataFilter: ajaxDataFilter
        },
        data: {
            simpleData: {
                enable: true
            }
        }
    };

    // 添加刷新按钮
    function addHoverDom(treeId, treeNode) {
        var aObj = $("#" + treeNode.tId + "_a");
        if(!treeNode.isParent){
            return;
        }
        if ($("#" + treeNode.tId + "_refresh").length > 0){
            return;
        }
        var refreshStr = $('<button type="button" class="icon-refresh" id="'+treeNode.tId+'_refresh" title="refresh" treenode_refresh=""></button >');
        aObj.append(refreshStr);
        refreshStr.bind("click", function(){
            var treeObj = $.fn.zTree.getZTreeObj(treeId);
            treeObj.reAsyncChildNodes(treeNode, "refresh");
        });
    };
    // 移除刷新按钮
    function removeHoverDom(treeId, treeNode) {
        $("#" + treeNode.tId + "_refresh").unbind().remove();
    };

    function beforeRemove(treeId, treeNode) {
        layer.confirm("是否删除？", function(index){
            function successFun(ret) {
                if (ret.code === 200) {
                    layer.msg("删除成功", {icon: 1});
                    var treeObj = $.fn.zTree.getZTreeObj(treeId);
                    treeObj.reAsyncChildNodes(treeNode.getParentNode(), "refresh");
                } else {
                    layer.msg(ret.msg, {icon: 2});
                }
            }
            var option = {
                type: 'get',
                url: api.plate.removeDirOrFile,
                success: successFun,
                data: {"fileName": treeNode.filePath}
            };
            utils.ajax(option);
            layer.close(index);
        });
        return false;
    }


    function beforeRename(treeId, treeNode, newName, isCancel) {
        function successFun(ret) {
            if (ret.code === 200) {
                var treeObj = $.fn.zTree.getZTreeObj(treeId);
                treeObj.reAsyncChildNodes(treeNode.getParentNode(), "refresh");
            } else {
                layer.msg(ret.msg, {icon: 2});
            }
        }
        var option = {
            type: 'get',
            url: api.plate.renameDirOrFile,
            success: successFun,
            data: {"fileName": treeNode.filePath, "newName": newName}
        };
        utils.ajax(option);

        var treeObj = $.fn.zTree.getZTreeObj(treeId);
        treeObj.refresh(treeNode);
        return false;
    }


    var isFirst = false;
    function onAsyncSuccess(event, treeId) {
        if (isFirst) {
            //获得树形图对象
            var treeObj = $.fn.zTree.getZTreeObj(treeId);
            var nodes = treeObj.getNodes();
            if (nodes.length>0) {
                for(var i=0;i<nodes.length;i++){
                    if(nodes[i].isParent){
                        treeObj.expandNode(nodes[i], true, false, false); // 展开第一层级
                    }
                }
            }
            isFirst= false;
        }
    }

    // 异步加载树结构数据
    function ajaxDataFilter(treeId, parentNode, ret) {
        var treeNode = [];
        if (ret.code === 200) {
            $.each(ret.obj, function (index, item){
                var node = {};
                node.id = item.id;
                node.name = item.fileName;
                node.isParent = item.isDir;
                node.filePath = encodeURI(item.filePath);   // 路径编码，防止出现特殊字符影响
                treeNode.push(node);
            });
        }
        return treeNode;
    };

    // 树节点点击事件
    function treeClick(event, treeId, node) {
        var treeObj = $.fn.zTree.getZTreeObj(treeId);

        if(node.name.indexOf(".png") > 1 || node.name.indexOf(".jpg") > 1){
            // $('#baseImage').attr("src", encodeURI(api.file.readFile + "?filePath=" + node.filePath));
            var img = new Image();
            img.src = encodeURI(api.file.readFile + "?filePath=" + node.filePath);

            setTimeout(function () {
                $("#clos").val(img.width);
                $("#rows").val(img.height);
                /*$("#c_clos").val($("#canvas").width());
                $("#c_rows").val($("#canvas").height());*/

                c.width = img.width;
                c.height = img.height;
                ctxt.drawImage(img,0, 0, img.width, img.height);

            }, 500);

            initSilder('50%');
        }

        if(node.isParent){
            $("#parentDir").val(node.filePath);
            dirTreeNode = node;
        } else {
            plateTreeNode = node;
        }
    }

    return {
        "init": init
    }
});