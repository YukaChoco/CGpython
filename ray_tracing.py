# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import numbers
import sys

'''
*******************************************

課題と書かれている場所のみを変更すること。
新たなライブラリ等をインポートすることは禁止する。

このソースコードは、Anaconda の base 環境で実行可能である。
Anaconda をインストールし、Anaconda Prompt あるいは Spyder を起動すれば実行できる。

もし、自分のPCで Anaconda の環境を構築できない場合は、以下の方法をとれ。

１． RAINBOW リモートサーバーにログインし、Anaconda 環境を利用する。
    $ ssh user_name@remote.ritsumei.ac.jp
    $ pyenv versions    （バージョン名を確認する）
    $ pyenv local anaconda3-2020.07  （使えるバージョンを指定する）
	$ python ray_tracing.py
	
	** ファイルの転送（アップロード・ダウンロード）には scp コマンドを使え。
    
２． Python 環境をインストールし、以下のパッケージをインストールする。
    $ pip install numpy pillow
    
    
便利な機能
    a, b をベクトル (np.ndarray) とする。
    ベクトルの足し算: sum = a + b
    ベクトルの内積： dot_product = np.dot(a, b)
    ベクトルの外積： cross_product = np.cross(a, b)
    ベクトルの要素ごとの積： element_wise_product = a * b
    ベクトルの絶対値（大きさ、L2ノルム）： length = np.linalg.norm(a)
    ３次元ベクトルを 0 で初期化： z = np.zeros(3)
    
    v を float （スカラー） とする。
    平方根： sqrt = np.sqrt(v)
    三角関数： s = np.sin(v), c=np.cos(v), t = np.tan(v)

*******************************************
'''

def ray_casting(i, j, fov, pixels):
    '''
    課題１：レイキャスティング
    
    Parameters
    ----------
    i : int
        i 番目のピクセル。整数値。画像のX座標の値。
    j : int
        j 番目のピクセル。整数値。画像のY座標の値。
    fov : float
        Field of View (FOV)の値。画角。上から下までの視野角。
    pixels : intzz
        画像の縦横のピクセル数。画像は正方形で、横 pixels ピクセル、縦 pixels ピクセルである。
        
    Returns
    -------
    ray : Ray
        キャストされた光線
        Ray クラスの初期化は、 Ray(p0, direction) である。p0 は光線の出どころ、direction は光線の方向ベクトルである。
        方向ベクトルは大きさを1に正規化しなくてもよい（内部で自動で正規化される）。
        p0 に 0 を指定すると、自動で (0, 0, 0) の座標に変換される。
        
    Notes
    -----
    カメラはピンホールカメラで、画像の中心をカメラの光軸が貫くものとする。また、世界座標系とカメラ座標系は同一とする。
	すなわち、画像平面は XY 平面と平行であり、カメラの視線方向は +Z 軸方向である。
    '''

    z = pixels / (2 * np.tan(fov / 2))
    direction = np.array([i - pixels/2, j- pixels/2, z])
    return Ray(0, direction)



def ray_sphere_intersection(ray, sphere):
    '''
    課題２：光線と球の衝突判定
    
    Parameters
    ----------
    ray : Ray
        判定を行う光線。光線の出どころは、ray.p0 (3次元ベクトル)、光線の方向ベクトルは、ray.direction (3次元ベクトル)で取得できる。
    sphere : Sphere
        判定を行う球。球の中心座標は、sphere.center （3次元ベクトル)、球の半径は、sphere.radius （スカラー）で取得できる。
    
    Returns
    -------
    t : float
        衝突点のカメラからの距離 (depth)であり、レイの直線方程式を媒介変数表示したときの t の値。衝突しない場合は負の値を返すこと。
        
    Notes
    -----
        a, b を３次元ベクトルとしたとき、ベクトルの内積は np.dot(a, b) で計算できる。
    '''

    # 使用するクラスの定義
    class SolutionAndCount:
            def __init__(self,solution_count, solutions):
                self.solution_count = solution_count
                self.solutions = solutions

    # 使用する関数の定義
    def solve_quadratic(a: float, b: float, c: float):
        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            return SolutionAndCount(0, [])

        elif discriminant == 0:
            x = -b / (2 * a)
            return SolutionAndCount(1, [x])

        else:
            sqrt_discriminant = np.sqrt(discriminant)
            x1 = (-b + sqrt_discriminant) / (2 * a)
            x2 = (-b - sqrt_discriminant) / (2 * a)
            return SolutionAndCount(2, [x1, x2])

    def choose_distance_from_solution(solution: SolutionAndCount):
        solution_count = solution.solution_count
        solutions = solution.solutions

        if solution_count == 0:
            return -1

        elif solution_count == 1:
            if solutions[0] > 0:
                return solutions[0]
            else:
                return -1

        else:
            x1 = solutions[0]
            x2 = solutions[1]
            if x1 > 0:
                if x2 > 0:
                    return min(x1, x2)
                else:
                    return x1
            elif x2 > 0:
                return x2
            return -1

    # 二次方程式の係数を算出
    p0_c = (ray.p0 - sphere.center)
    a = np.dot(ray.direction, ray.direction)
    b = 2 * np.dot(ray.direction, p0_c)
    c = (np.dot(p0_c, p0_c)) - (sphere.radius * sphere.radius)

    # 係数・関数を用いて距離を算出
    solution = solve_quadratic(a, b, c)
    t = choose_distance_from_solution(solution)

    return t


def ray_triangle_intersection(ray, triangle):
    n = triangle.normal
    denom = np.dot(ray.direction, n)
    if denom == 0:
		# 平面と光線が平行なため衝突しない。
        return -1
    t = (np.dot(triangle.A, n) - np.dot(ray.p0, n)) / denom
    if t < 0:
		# 平面と衝突しない。（カメラより後ろ側に平面がある）
        return -1
    p = ray.p0 + t * ray.direction
    ca = np.cross(triangle.A - triangle.C, p - triangle.A)
    cb = np.cross(triangle.B - triangle.A, p - triangle.B)
    cc = np.cross(triangle.C - triangle.B, p - triangle.C)
    ab = np.dot(ca, cb)
    bc = np.dot(cb, cc)
	# 外積の向きが3つとも同じかどうかを判定
    if ab > 0 and bc > 0:
		# 三角形の内側
        return t
	# 三角形の外側
    return -1


def find_intersect(ray, scene):
	# 最も近い衝突点までの距離と、その物体のインスタンスを返す。
    depth = sys.float_info.max
    nearest = None
    for obj in scene.objects:
		# オブジェクトごとに衝突判定をして、距離を求める。
        if isinstance(obj, Sphere):
            d = ray_sphere_intersection(ray, obj)
        if isinstance(obj, Triangle):
            d = ray_triangle_intersection(ray, obj)
        if d < 0:
            continue
		# 近い衝突点が見つかれば更新
        if d < depth:
            depth = d
            nearest = obj
    return depth, nearest



def find_color(ray, lights, obj, depth, scene):
    if obj is None:
        return np.zeros(3) # Black
    ''' 
    課題４：（CG応用ではオプション）影の実装 (Visibility の実装)
    物体表面から光源の間に物体が存在する場合は影（色を黒にする）となるようにせよ。
    この課題では、関数の引数を変更する必要がある（影の判定にはシーン全体のオブジェクト scene が必要）。
    影の判定には find_intersect(ray, scene) 関数を使用するとよい。
    '''
    if isinstance(obj, Sphere):
        '''
        課題３：球の表面の色を計算。 Visibility は考慮しなくてよい (=1) 。
        
        Parameters
        ----------
        ray : Ray
            計算を行う光線。光線の出どころは ray.p0, 光線の方向は、ray.direction である。
        lights : 光源オブジェクト （Light クラスのインスタンス） の配列
            シーン中のすべての光源
        obj : Sphere (find_color 自体の obj は何でもよいが、isinstance で Sphere と判定している。)
            計算を行う球。球の BRDF の値は、obj.brdf.reflectance(light_direction, viewing, normal) で取得できる。引数はそれぞれ、物体表面から見た光源方向・視線方向・法線方向で全て大きさは１。
        depth : float
            衝突判定をしたときに返した、カメラと衝突点との距離。
            
        Returns
        -------
        color : 3次元ベクトル
            物体表面の色。R, G, B の順で、大きさは 0 から 1 の間の小数。
            
        Notes
        -----
            光源の色は、light.color, 光源が点p に届いた時の光の減衰は light.attenuation(p) で計算できる。 light は 引数 lights の i 番目の光源である。
            右目と左目の色はサンプルと入れ替わっていてもよい （これは、右手系・左手系が未定義なことにより、どちらの座標系で計算しているかによって左右反転が起きるからである）。
        '''

        position = ray.p0 + depth * ray.direction # 衝突点の座標
        normal =  (position - obj.center) / obj.radius # 法線方向
        viewing = -1. * ray.direction # 視線方向
        color = np.zeros(3) # 色を黒で初期化
    
        for light in lights:
            light_direction = light.direction(position) # 光源方向

            visible_position = ray.p0 + (depth - 0.1) * ray.direction # 衝突点の座標(カメラ側に少しずらす)
            light_ray = Ray(visible_position, light_direction) # 光源方向へのレイを定義
            _collision_depth, collision_obj = find_intersect(light_ray, scene) # 光源・衝突点間の衝突判定
            if collision_obj is None : # Visibility = 1 の場合
                color += obj.brdf.reflectance(light_direction, viewing, normal) * light.attenuation(position) * light.color

        return color

    if isinstance(obj, Triangle):
        # 3角形の色を計算
        p = ray.p0 + depth * ray.direction
        n = obj.normal # normal
        v = -1. * ray.direction # viewing
        c = np.zeros(3) # 色を黒で初期化
        for light in lights:
            # 光源が複数ある場合は、各光源ごとに色を計算し足し合わせる
            l = light.direction(p) # light_direction
            c += obj.brdf.reflectance(l, v, n) * light.attenuation(p) * light.color
        return c # 3角形の色
    return np.zeros(3) # Black
    


'''
*******************************************

「CG 応用」 の課題（課題 1 ～ 3 のみ）においては、これより下のソースコードは変更しなくてよい。

*******************************************
'''


def ray_trace(scene):
    ''' 課題５：カメラのパラメータ（位置姿勢・焦点距離）が変わった時に対応する。
    
    scene.camera_intrinsic_param がカメラの内部パラメータ（焦点距離など）
    scene.camera_extrinsic_param がカメラの外部パラメータ（カメラの位置と回転（姿勢））
    scene.camera_pixels がカメラの画素数
    上記のパラメータを使って、下記のコードを書き換えること。
    
    また、この変更が確認できるように、カメラの位置姿勢を変えたものをいくつかレンダリングせよ。
    例えば、__main__() において、Scene クラスの dolly_and_pan() を使うと、カメラの位置姿勢が少し変わる。
    '''
    # カメラの設定は、以下のとおりとする。（camera_intrinsic_param を無視して固定値を使用している。課題5では書き換えよ）
    pixels = scene.camera_pixels[0] # ピクセル数
    f = scene.camera_intrinsic_param[0, 0]
    fov = 2 * np.arctan( pixels / (2 * f) ) # ラジアン

    ex_param = scene.camera_extrinsic_param
    translate_param = ex_param[:, 3]
    rotate_param = ex_param[:3, :3]

    # レンダリング開始
    image = np.zeros((pixels, pixels, 3))
    for u in range(pixels):
        for v in range(pixels):
            # 課題5では、fov, u, v, pixels ともに camera_intrinsic_param から求める値を用いよ。
            default_ray = ray_casting(u, v, fov, pixels)
            ''' 課題5では、ここにコードを追加する必要がある。
            上記 ray はカメラ座標系で表現されているので、カメラの位置姿勢 (camera_extrinsic_param) に応じて世界座標系における表現に変換せよ。
            光線の始点 ray.p0 は平行移動の影響を受け、光線の方向 ray.direction は回転の影響を受ける。
            '''
            ray = Ray(default_ray.p0 + translate_param, np.dot(rotate_param, default_ray.direction))
            depth, obj = find_intersect(ray, scene)
            ''' 課題４では、find_color に scene を渡すように変更しないと、影の実装はできない。
            '''
            color = find_color(ray, scene.lights, obj, depth, scene)
            image[v, u] = color
            
    # 結果を保存
    image = np.uint8(np.minimum(image * 255., 255))
    image = Image.fromarray(image)
    image.save('result.png')
    


def to_3d_array(val):
	# 3次元ベクトルであることをチェック。変換できる場合は変換する。
    if isinstance(val, numbers.Number):
        val = np.ones(3) * val
    if val is None:
        val = np.zeros(3)
    if (not isinstance(val, np.ndarray)) and hasattr(val, "__iter__"):
        val = np.array(val)
    assert isinstance(val, np.ndarray)
    assert len(val.shape) == 1
    assert val.shape[0] == 3
    return val


def vec_normalize(val):
	# ノルムが1の3次元ベクトルにする。
    val = to_3d_array(val)
    size = np.linalg.norm(val)
    assert size > 0
    return val / size
    


class Ray:
    def __init__(self, p0, direction):
        self.set_p0(p0)
        self.set_direction(direction)
        
        
    def set_p0(self, p0):
        self._p0 = to_3d_array(p0)
    
    def set_direction(self, direction):
        self._direction = vec_normalize(direction)
        
    @property
    def p0(self):
        return self._p0
    @p0.setter
    def p0(self, val):
        self.set_p0(val)
    
    @property
    def direction(self):
        return self._direction
    @direction.setter
    def direction(self, val):
        self.set_direction(val)
        

class BRDF:
    def reflectance(self, lighting, viewing, normal):
        return np.ones(3)
    

class Diffuse(BRDF):
    def __init__(self, diffuse_color):
        self.color = to_3d_array(diffuse_color)
        
    def reflectance(self, lighting, viewing, normal):
        dot = max(0., np.dot(lighting, normal))
        return self.color * dot
        
    

    
class Phong(BRDF):
    ''' 課題６： Phong のモデルを実装せよ。 reflectance も合わせて変更する必要がある。
    また、Scene クラスや __main__ において Phong の反射モデルを使用するようにシーンを作成してレンダリングせよ。
    例えば、set_default_scene において、Diffuse(...) と書かれている部分を Phong(...) に置き換えると、物体の反射モデルを変更できる。
    '''
    def __init__(self, diffuse_color, specular_color, alpha):
        self._diffuse_color = diffuse_color
        self._specular_color = specular_color
        self._alpha = alpha
        raise NotImplementedError()
        
    def reflectance(self, lighting, viewing, normal):
        raise NotImplementedError()
    
        
class Light:
    def direction(self, scene_point):
        return np.array((0., 0., 1.))
    
    def attenuation(self, scene_point):
        return 1.
    
    @property
    def color(self):
        return np.ones(3)
    
    
class ParallelLight(Light):
    def __init__(self, direction, power=1.):
        self._d = direction
        self._color = to_3d_array(power)
        
    def direction(self, scene_point):
        return self._d
    
    @property
    def color(self):
        return self._color
    
class PointLight(Light):
    def __init__(self, position, power=1.):
        self._p = position
        self._color = to_3d_array(power)
        
    def direction(self, scene_point):
        return vec_normalize(self._p - scene_point) # if this is assertion error, the scene is bad (reflection point and point light source is at the same position.)
    
    def attenuation(self, scene_point):
        dist = np.linalg.norm(self._p - scene_point)
        assert dist > 0 , "if this is failed, the scene is bad (reflection and light source points may be the same position.)"
        return 1. / (dist*dist)
    
    @property
    def color(self):
        return self._color
    
        
class Sphere:
    def __init__(self, center, radius, brdf):
        self.center = center
        self.radius = radius
        self.brdf = brdf
        
    @property
    def center(self):
        return self._center
    @center.setter
    def center(self, val):
        self._center = to_3d_array(val)
        
    @property
    def radius(self):
        return self._radius
    @radius.setter
    def radius(self, val):
        assert isinstance(val, numbers.Number)
        self._radius = float(val)
        
        
class Triangle:
    def __init__(self, a, b, c, brdf):
        self.A = a
        self.B = b
        self.C = c
        self.brdf = brdf
        
    @property
    def A(self):
        return self._A
    @A.setter
    def A(self, val):
        self._A = to_3d_array(val)        
        
    @property
    def B(self):
        return self._B
    @B.setter
    def B(self, val):
        self._B = to_3d_array(val)
        
    @property
    def C(self):
        return self._C
    @C.setter
    def C(self, val):
        self._C = to_3d_array(val)
        
    @property
    def normal(self):
        n = np.cross(self.C - self.A, self.B - self.A)
        return n / np.linalg.norm(n)
    
    
class Scene:
    def __init__(self):
        self.lights = []
        self.objects = []
        self.camera_pixels = ((300, 300))
        f = 100. / np.tan(20. * np.pi / 180.)
        self.camera_extrinsic_param = np.array(((1., 0., 0., 0.),
                                                (0., 1., 0., 0.),
                                                (0., 0., 1., 0.)))
        self.camera_intrinsic_param = np.array(((f, 0., 100. ),
                                                (0., f, 100. ),
                                                (0., 0., 1.  )))
        
    def dolly_and_pan(self, translate, rotate_y):
        theta = rotate_y * np.pi / 180.
        self.camera_extrinsic_param = np.array(((np.cos(theta), 0., 0.-np.sin(theta), translate[0]),
                                                (0., 1., 0., translate[1]),
                                                (np.sin(theta), 0., np.cos(theta), translate[2])))
        
    def set_defalut_scene(self):
		# シーンに光源を追加する。
        self.lights.append(ParallelLight(vec_normalize((-1., -1., -2)))) # 平行光源
        self.lights.append(PointLight(np.array((0., 0., 0.)), 1000.)) # 点光源
		
		# シーンに物体を追加する。プリミティブと反射モデルを設定する。
        self.objects.append(Sphere((0, 0, 50), 12, Diffuse((1., 0., 0.)))) # 顔全体
        self.objects.append(Sphere((-1.5, -2, 20), 1.3, Diffuse((0., .6, .9)))) #目-緑
        self.objects.append(Sphere((1.5, -2, 20), 1.3, Diffuse((0., .8, .3)))) #目-青
        self.objects.append(Triangle((-10., -4., 42.), (10., -4., 42.), (0., 10, 38.), Diffuse((0.9, 1., 0.)))) # 黄緑の三角形
        self.objects.append(Sphere((0, 0, -30), 10, Diffuse((0., 1., 0.)))) # カメラの裏側にある球 - 青 （デバック用）
        
		

def camera_move_animation():
	# 画像を複数レンダリングして、アニメーションGIF として保存。
    scene = Scene()
    scene.set_defalut_scene()
    imgs = []
    for x in range(3):
		# カメラの位置姿勢を変更
		# scene.camera_extrinsic_param = ...
        imgs.append(ray_trace(scene))
    imgs[0].save('result.gif', save_all=True, append_images = imgs, loop=0)
	
	
if __name__ == '__main__':
    scene = Scene()
    scene.set_defalut_scene()
    scene.dolly_and_pan(np.array((0, 0, 0)), 0)
    ray_trace(scene)