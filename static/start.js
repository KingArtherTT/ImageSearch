function get_result(img_name) {
    $('#result').css('display', 'block')
    // 发起request 请求，返回四张图片的文件名称
    var data={}
    data['img_name'] = img_name
    $.ajax({
        type: 'POST',
        url: '/action/get_result/',
        data: JSON.stringify(data),
        dataType: 'json',
        contentType: 'application/json; charset=UTF-8',
        success: function (result) {
            console.log(result)
            if (parseInt(result['code']) == 1) {
                //替换照片
                $('#result .col-md-3').each(function (i, e) {
                    $(e).find('img').attr('src', '/static/img_data/oxbuild_images/' + result['img_names'][i])
                    $(e).find('img').attr('alt', result['img_names'][i])
                    $(e).find('img').css('width', '287px')
                    $(e).find('.caption').html(result['img_names'][i])
                })
            } else {
                console.log(result['message'])
                alert(result['message'])
            }
        }
    })
}

function a_click(){
    img_name = $(this).children('img').attr('alt')
    get_result(img_name)
}

function refresh_search(){
    $.ajax({
        type: 'POST',
        url: '/action/change_search/',
        // data: JSON.stringify(data),
        dataType: 'json',
        contentType: 'application/json; charset=UTF-8',
        success: function (result) {
            console.log(result)
            //替换照片
            $('#search .col-md-3').each(function (i, e) {
                $(e).find('img').attr('src', '/static/img_data/oxbuild_images/' + result[i])
                $(e).find('img').attr('alt', result[i])
                $(e).find('img').css('width', '287px')
                $(e).find('.caption').html(result[i])
            })
        }
    })
}

$(document).ready(function () {
    $('#search span').each(function (i, e) {
        $(e).on('click',a_click)
    })
    $('#refresh').on('click',refresh_search)
    refresh_search()
})