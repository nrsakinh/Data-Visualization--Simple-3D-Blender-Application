import sys
import vtk
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QUndoStack, QUndoCommand
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFrame, QToolButton, QStyle
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QStyleFactory
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util.colors import cornflower
import os

ICON_DIR = r"D:\SDV2025\project_env1\SDV_Assignment1_Sakinah_Ezyan_Sureka_Charlene\Icons"

def _icon(filename, fallback_style=None, fallback_enum=None):
    """
    Return QIcon for a file inside ICON_DIR; fallback to a Qt standard icon if missing.
    Example: _icon('sphere_icon.png', self.style(), QStyle.SP_FileIcon)
    """
    path = os.path.join(ICON_DIR, filename)
    if os.path.isfile(path):
        ic = QtGui.QIcon(path)
        if not ic.isNull():
            return ic
    if fallback_style and fallback_enum is not None:
        return fallback_style.standardIcon(fallback_enum)
    return QtGui.QIcon()

class myVTK:
    """
    Core VTK logic class. Now includes file loading capabilities
    """
    def __init__(self):
        self.colors = vtk.vtkNamedColors()
        self.actors = []
        self.mappers = []
        self.sources = []
        self.renderer = None
        self.window = None
        self.interactor = None
        self.current_object_name = 'sphere'
        self.current_color = (1.0, 1.0, 1.0)
        self.axes_widget = None  # Add this line
        self.grid_actor = None  # Add this line
        self.axis_actors = []   # Add this line
        self.lights = []
        # render lifecycle flag
        self._alive = True
        self.click_observer = None
        self.axis_release_observer = None  # NEW
        self._axis_click_active = False
        self._saved_style = None

    def get_reader_for_file(self, file_path):
        """Determines the appropriate VTK reader based on file extension."""
        extension = os.path.splitext(file_path)[1].lower()
        if extension == ".stl":
            return vtk.vtkSTLReader()
        elif extension == ".obj":
            # OBJ is handled separately in load_file
            return vtk.vtkOBJReader()
        elif extension == ".ply":
            return vtk.vtkPLYReader()
        elif extension in [".vtk", ".vtp"]:
            # vtkGenericDataObjectReader can handle legacy and XML VTK formats
            return vtk.vtkGenericDataObjectReader()
        else:
            print(f"Unsupported file format: {extension}")
            return None

    def load_file(self, file_path):
        """Loads a model from a file and returns (actor, object_name)."""
        reader = self.get_reader_for_file(file_path)
        if not reader:
            return None, None

        extension = os.path.splitext(file_path)[1].lower()
        object_name = os.path.basename(file_path)

        if extension == ".obj":
            # Use OBJ reader (handles mtllib/usemtl)
            obj_reader = vtk.vtkOBJReader()
            obj_reader.SetFileName(file_path)
            obj_reader.Update()
            poly_data = obj_reader.GetOutput()

            clean = vtk.vtkCleanPolyData()
            clean.SetInputData(poly_data)
            clean.Update()

            # Wrap in a producer for the mapper function
            tp = vtk.vtkTrivialProducer()
            tp.SetOutput(clean.GetOutput())
            mapper = self.create_mapper(tp)
            actor = self.create_actor(mapper)
            self.orient_actor_y_up_to_z_up(actor)

            # If OBJ already has UVs keep them; else auto planar
            if not poly_data.GetPointData().GetTCoords():
                self._ensure_texture_coordinates(actor)

            # Brighter defaults for textured models
            prop = actor.GetProperty()
            prop.SetAmbient(0.3)
            prop.SetDiffuse(0.8)
            prop.SetSpecular(0.2)

            return actor, object_name
        else:
            reader.SetFileName(file_path)
            reader.Update()
            mapper = self.create_mapper(reader)
            actor = self.create_actor(mapper)
            object_name = os.path.basename(file_path)
            return actor, object_name
    
    def load_3ds_scene(self, file_path):
        """
        Minimal custom 3DS loader: vertices, faces, UVs, multi-object support.
        Ignores empty nodes. Returns list of (actor, name).
        """

        import struct

        def read_chunk(f):
            """Reads (chunk_id, chunk_length, chunk_start)."""
            data = f.read(6)
            if len(data) < 6:
                return None, None, None
            cid, length = struct.unpack("<HI", data)
            return cid, length, f.tell()

        def skip_to(f, chunk_start, length):
            """Skip to end of chunk."""
            f.seek(chunk_start + length - 6)

        def read_cstring(f):
            """Read zero-terminated string."""
            s = b""
            while True:
                c = f.read(1)
                if c == b"" or c == b"\x00":
                    break
                s += c
            return s.decode("utf-8", errors="ignore")

        # Store meshes here
        meshes = {}   # name → {verts:[], faces:[], uvs:[]}

        with open(file_path, "rb") as f:
            # Root chunk
            cid, length, pos = read_chunk(f)
            if cid != 0x4D4D:   # MAIN3DS
                print("Not a valid .3ds file")
                return []

            while f.tell() < length:
                cid, clen, start = read_chunk(f)
                if not cid:
                    break

                # EDIT3DS
                if cid == 0x3D3D:
                    end = start + (clen - 6)
                    while f.tell() < end:
                        scid, sclen, sstart = read_chunk(f)
                        if not scid:
                            break

                        # Object block
                        if scid == 0x4000:
                            name = read_cstring(f)
                            meshes[name] = {
                                "verts": [],
                                "faces": [],
                                "uvs": []
                            }

                            # Read object sub-chunks
                            while f.tell() < sstart + (sclen - 6):
                                ocid, oclen, ostart = read_chunk(f)
                                if not ocid:
                                    break

                                # Mesh block
                                if ocid == 0x4100:
                                    # Inside mesh block
                                    mend = ostart + (oclen - 6)
                                    while f.tell() < mend:
                                        mcid, mclen, mstart = read_chunk(f)
                                        if not mcid:
                                            break

                                        # VERTEX LIST
                                        if mcid == 0x4110:
                                            vcount = struct.unpack("<H", f.read(2))[0]
                                            for _ in range(vcount):
                                                x, y, z = struct.unpack("<fff", f.read(12))
                                                meshes[name]["verts"].append((x, y, z))

                                        # FACE LIST
                                        elif mcid == 0x4120:
                                            fcount = struct.unpack("<H", f.read(2))[0]
                                            for _ in range(fcount):
                                                a, b, c, flag = struct.unpack("<HHHH", f.read(8))
                                                meshes[name]["faces"].append((a, b, c))

                                        # UV LIST
                                        elif mcid == 0x4140:
                                            tcount = struct.unpack("<H", f.read(2))[0]
                                            for _ in range(tcount):
                                                u, v = struct.unpack("<ff", f.read(8))
                                                meshes[name]["uvs"].append((u, 1 - v))

                                        skip_to(f, mstart, mclen)

                                skip_to(f, ostart, oclen)

                        skip_to(f, sstart, sclen)

                skip_to(f, start, clen)

        # ----------- Build VTK actors -----------
        output = []

        for name, data in meshes.items():
            verts = data["verts"]
            faces = data["faces"]
            uvs = data["uvs"]

            if len(verts) == 0 or len(faces) == 0:
                continue  # ignore empty nodes, per your Option A

            # Build vtkPolyData
            pts = vtk.vtkPoints()
            for v in verts:
                pts.InsertNextPoint(v)

            polys = vtk.vtkCellArray()
            for a, b, c in faces:
                polys.InsertNextCell(3)
                polys.InsertCellPoint(a)
                polys.InsertCellPoint(b)
                polys.InsertCellPoint(c)

            poly = vtk.vtkPolyData()
            poly.SetPoints(pts)
            poly.SetPolys(polys)

            # UVs
            if len(uvs) == len(verts):
                tcoords = vtk.vtkFloatArray()
                tcoords.SetNumberOfComponents(2)
                tcoords.SetName("TextureCoordinates")
                for uv in uvs:
                    tcoords.InsertNextTuple(uv)
                poly.GetPointData().SetTCoords(tcoords)

            # Wrap in VTK pipeline
            tp = vtk.vtkTrivialProducer()
            tp.SetOutput(poly)
            mapper = self.create_mapper(tp)
            actor = self.create_actor(mapper)

            output.append((actor, name))

        print(f"✓ Imported {len(output)} mesh object(s) from 3DS")
        return output


    def create_parametric(self, kind: str):
        mapping = {
            "torus": vtk.vtkParametricTorus(),
            "super_ellipsoid": vtk.vtkParametricSuperEllipsoid(),
            "klein": vtk.vtkParametricKlein(),
            "mobius": vtk.vtkParametricMobius()
        }
        func = mapping.get(kind)
        if not func:
            print(f"Unknown parametric surface: {kind}")
            return None, None
        src = vtk.vtkParametricFunctionSource()
        src.SetParametricFunction(func)
        src.SetUResolution(64)
        src.SetVResolution(64)
        src.Update()
        mapper = self.create_mapper(src)
        actor = self.create_actor(mapper)
        base_name = f"param_{kind}"
        self.current_object_name = base_name
        print(f"✓ Created parametric surface: {base_name}")
        return actor, base_name

    def load_texture(self, texture_path):
        """Load a texture from an image file."""
        extension = os.path.splitext(texture_path)[1].lower()
        
        if extension in ['.jpg', '.jpeg']:
            reader = vtk.vtkJPEGReader()
        elif extension == '.png':
            reader = vtk.vtkPNGReader()
        elif extension in ['.bmp']:
            reader = vtk.vtkBMPReader()
        else:
            print(f"Unsupported texture format: {extension}")
            return None
        
        if not reader.CanReadFile(texture_path):
            print(f"Cannot read texture file: {texture_path}")
            return None
        
        reader.SetFileName(texture_path)
        reader.Update()
        
        texture = vtk.vtkTexture()
        texture.SetInputConnection(reader.GetOutputPort())
        texture.InterpolateOn()
        # Avoid tiling when UVs go beyond [0,1]
        texture.RepeatOff()
        try:
            texture.EdgeClampOn()
        except Exception:
            pass
        return texture
    
    def orient_actor_y_up_to_z_up(self, actor):
        """Convert Y-up models (OBJ/3DS) to this app's Z-up world by rotating +90° about X."""
        if not actor:
            return
        actor.RotateX(90.0)
    
    def create_implicit_object(self, object_type):
        """Creates an iso-surface from an implicit function (sample + contour + normals)."""
        print(f"Creating implicit {object_type}...")
        if object_type == 'quadric_sphere':
            quadric = vtk.vtkQuadric()
            quadric.SetCoefficients(1, 1, 1, 0, 0, 0, 0, 0, 0, -16)  # x^2+y^2+z^2-4^2=0
            sample = vtk.vtkSampleFunction()
            sample.SetImplicitFunction(quadric)
            sample.SetSampleDimensions(64, 64, 64)
            sample.SetModelBounds(-5, 5, -5, 5, -5, 5)
            contour = vtk.vtkContourFilter()
            contour.SetInputConnection(sample.GetOutputPort())
            contour.GenerateValues(1, 0.0, 0.0)  # iso-value = 0

        elif object_type == 'torus':
            superq = vtk.vtkSuperquadric()
            superq.SetToroidal(1)
            superq.SetSize(4.0)        # major radius
            superq.SetThickness(0.5)   # minor/ tube radius relative to size
            superq.SetThetaRoundness(1.0)
            superq.SetPhiRoundness(1.0)
            sample = vtk.vtkSampleFunction()
            sample.SetImplicitFunction(superq)
            sample.SetSampleDimensions(128, 128, 64)
            sample.SetModelBounds(-8, 8, -8, 8, -4, 4)
            contour = vtk.vtkContourFilter()
            contour.SetInputConnection(sample.GetOutputPort())
            contour.GenerateValues(1, 0.0, 0.0)

        else:
            print(f"Unknown implicit type: {object_type}")
            return None, None

        # Update to produce geometry
        contour.Update()

        # Mapper via unified helper (adds normals, disables scalar coloring)
        mapper = self.create_mapper(contour)

        actor = self.create_actor(mapper)
        self.current_object_name = object_type
        print(f"✓ {object_type} implicit surface created")
        return actor, object_type
        
    def create_cell_object(self, cell_type):
        """Creates a programmatic object based on a specific cell type."""
        if cell_type == 'convex_point_set':
            print("Creating Convex Point Set object...")
            points = vtk.vtkPoints()
            points.InsertNextPoint(0, 0, 0)
            points.InsertNextPoint(1, 0, 0)
            points.InsertNextPoint(1, 1, 0)
            points.InsertNextPoint(0, 1, 0)
            points.InsertNextPoint(0, 0, 1)
            points.InsertNextPoint(1, 0, 1)
            points.InsertNextPoint(1, 1, 1)
            points.InsertNextPoint(0, 1, 1)

            convexPointSet = vtk.vtkConvexPointSet()
            for i in range(8):
                convexPointSet.GetPointIds().InsertId(i, i)
            
            ugrid = vtk.vtkUnstructuredGrid()
            ugrid.SetPoints(points)
            ugrid.InsertNextCell(convexPointSet.GetCellType(), convexPointSet.GetPointIds())

            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputData(ugrid)
            actor = self.create_actor(mapper)
            return actor, "ConvexPointSet"

        if cell_type == 'polyhedron_cell':
            print("Creating Polyhedron (triangular frustum) cell...")

            # --- 1. Define points ---
            # Bottom triangle (z = 0)
            # Wide base so the taper is obvious
            points = vtk.vtkPoints()
            points.InsertNextPoint(-1.0, -1.0, 0.0)  # 0
            points.InsertNextPoint( 1.0, -1.0, 0.0)  # 1
            points.InsertNextPoint( 0.0,  1.0, 0.0)  # 2

            # Top smaller triangle (z = 1)
            # Same orientation, scaled toward the center
            scale = 0.5
            points.InsertNextPoint(-scale, -scale, 1.0)  # 3
            points.InsertNextPoint( scale, -scale, 1.0)  # 4
            points.InsertNextPoint( 0.0,   scale, 1.0)  # 5

            ugrid = vtk.vtkUnstructuredGrid()
            ugrid.SetPoints(points)

            # --- 2. Define faces as a vtkIdList ---
            # Layout: numFaces,
            #         (numPtsFace0, ids...),
            #         (numPtsFace1, ids...), ...

            faces = vtk.vtkIdList()

            # We have 5 faces:
            #   - bottom triangle
            #   - top triangle
            #   - 3 quad side faces
            faces.InsertNextId(5)

            # Bottom face (triangle): 0, 1, 2
            faces.InsertNextId(3)      # number of points in this face
            faces.InsertNextId(0)
            faces.InsertNextId(1)
            faces.InsertNextId(2)

            # Top face (triangle): 3, 4, 5
            faces.InsertNextId(3)
            faces.InsertNextId(3)
            faces.InsertNextId(4)
            faces.InsertNextId(5)

            # Side face 1 (quad): 0, 1, 4, 3
            faces.InsertNextId(4)
            faces.InsertNextId(0)
            faces.InsertNextId(1)
            faces.InsertNextId(4)
            faces.InsertNextId(3)

            # Side face 2 (quad): 1, 2, 5, 4
            faces.InsertNextId(4)
            faces.InsertNextId(1)
            faces.InsertNextId(2)
            faces.InsertNextId(5)
            faces.InsertNextId(4)

            # Side face 3 (quad): 2, 0, 3, 5
            faces.InsertNextId(4)
            faces.InsertNextId(2)
            faces.InsertNextId(0)
            faces.InsertNextId(3)
            faces.InsertNextId(5)

            # --- 3. Insert the polyhedron cell ---
            # (Using the same 2-argument style that worked for your cube)
            ugrid.InsertNextCell(vtk.VTK_POLYHEDRON, faces)

            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputData(ugrid)
            actor = self.create_actor(mapper)
            return actor, "PolyhedronCell"

        print(f"Unknown cell type: {cell_type}")
        return None, None
    
    def create_reduced_cube(self, object_id):
        """Creates a high-poly cube, then reduces its polygon count (subdivide + decimate)."""
        print("Creating Reduced Cube object...")

        # 1. Base cube source
        cube_source = vtk.vtkCubeSource()
        cube_source.SetXLength(8)
        cube_source.SetYLength(8)
        cube_source.SetZLength(8)

        # 2. Convert quads to triangles
        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputConnection(cube_source.GetOutputPort())

        # 3. Subdivide to create a high-poly cube
        subdivide = vtk.vtkLoopSubdivisionFilter()
        subdivide.SetInputConnection(triangle_filter.GetOutputPort())
        subdivide.SetNumberOfSubdivisions(2)  # try 1–3; higher = more triangles

        # 4. Decimate the dense mesh to reduce polygon count
        decimate = vtk.vtkDecimatePro()
        decimate.SetInputConnection(subdivide.GetOutputPort())
        decimate.SetTargetReduction(0.7)      # 0.7 = reduce ~70% of triangles
        decimate.PreserveTopologyOn()         # keep the cube closed
        decimate.BoundaryVertexDeletionOff()  # keep outer silhouette stable

        # 5. Mapper + actor
        mapper = self.create_mapper(decimate)  # expects GetOutputPort()
        actor = self.create_actor(mapper)

        self.current_object_name = 'reduced_cube'
        return actor, 'ReducedCube'

    def create_subdivided_cube(self, subdivisions):
        print("Creating Subdivided Cube object...")

        try:
            n = int(subdivisions)
        except Exception:
            n = 1
        if n < 1:
            n = 1

        # Target box size (matches the standard cube in this app)
        len_x = 8.0
        len_y = 8.0
        len_z = 8.0
        hx, hy, hz = len_x * 0.5, len_y * 0.5, len_z * 0.5

        # Build six planes with matching orientation so normals point outward
        faces = []

        def plane(origin, p1, p2):
            pl = vtk.vtkPlaneSource()
            pl.SetOrigin(*origin)
            pl.SetPoint1(*p1)
            pl.SetPoint2(*p2)
            pl.SetXResolution(n)
            pl.SetYResolution(n)
            return pl

        # +X
        faces.append(plane((+hx, -hy, -hz), (+hx, +hy, -hz), (+hx, -hy, +hz)))
        # -X
        faces.append(plane((-hx, +hy, -hz), (-hx, -hy, -hz), (-hx, +hy, +hz)))
        # +Y
        faces.append(plane((+hx, +hy, -hz), (-hx, +hy, -hz), (+hx, +hy, +hz)))
        # -Y
        faces.append(plane((-hx, -hy, -hz), (+hx, -hy, -hz), (-hx, -hy, +hz)))
        # +Z
        faces.append(plane((-hx, -hy, +hz), (+hx, -hy, +hz), (-hx, +hy, +hz)))
        # -Z
        faces.append(plane((-hx, +hy, -hz), (+hx, +hy, -hz), (-hx, -hy, -hz)))

        append = vtk.vtkAppendPolyData()
        for pl in faces:
            append.AddInputConnection(pl.GetOutputPort())
        append.Update()

        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(append.GetOutputPort())
        clean.PointMergingOn()
        clean.Update()

        # Create mapper/actor via the standard shaded pipeline (adds normals)
        mapper = self.create_mapper(clean)
        actor = self.create_actor(mapper)
        self.current_object_name = "SubdividedCube"
        print(f"✓ SubdividedCube created with {n} subdivisions per edge")
        return actor, "SubdividedCube"

    def create_object(self, object_type):
        print(f"Creating {object_type} object...")
        if object_type == 'sphere':
            source = vtk.vtkSphereSource()
            source.SetCenter(0, 0, 0)
            source.SetRadius(5.0)
            # Higher tessellation to see lighting gradients clearly
            source.SetThetaResolution(64)
            source.SetPhiResolution(64)
        elif object_type == 'cube':
            source = vtk.vtkCubeSource()
            source.SetXLength(8)
            source.SetYLength(8)
            source.SetZLength(8)
        elif object_type == 'cone':
            source = vtk.vtkConeSource()
            source.SetHeight(8.0)
            source.SetRadius(4.0)
            source.SetResolution(64)
            source.CappingOn()
        elif object_type == 'cylinder':
            source = vtk.vtkCylinderSource()
            source.SetHeight(8.0)
            source.SetRadius(4.0)
            source.SetResolution(64)
            source.CappingOn()
        elif object_type == 'pyramid':
            # Square pyramid (base 8x8 on Z=0, height 6 along +Z)
            pts = vtk.vtkPoints()
            pts.InsertNextPoint(-4.0, -4.0, 0.0)  # 0
            pts.InsertNextPoint( 4.0, -4.0, 0.0)  # 1
            pts.InsertNextPoint( 4.0,  4.0, 0.0)  # 2
            pts.InsertNextPoint(-4.0,  4.0, 0.0)  # 3
            pts.InsertNextPoint( 0.0,  0.0, 6.0)  # 4 apex

            polys = vtk.vtkCellArray()
            def tri(a,b,c):
                cell = vtk.vtkTriangle()
                cell.GetPointIds().SetId(0,a)
                cell.GetPointIds().SetId(1,b)
                cell.GetPointIds().SetId(2,c)
                polys.InsertNextCell(cell)

            # sides
            tri(0,1,4); tri(1,2,4); tri(2,3,4); tri(3,0,4)
            # base (two triangles)
            tri(0,1,2); tri(0,2,3)

            pd = vtk.vtkPolyData()
            pd.SetPoints(pts)
            pd.SetPolys(polys)

            tp = vtk.vtkTrivialProducer()
            tp.SetOutput(pd)
            source = tp
        elif object_type == 'rectangle':
            # 8x8 rectangle in XY plane at Z=0
            plane = vtk.vtkPlaneSource()
            plane.SetOrigin(-4.0, -4.0, 0.0)
            plane.SetPoint1( 4.0, -4.0, 0.0)
            plane.SetPoint2(-4.0,  4.0, 0.0)
            plane.SetXResolution(1)
            plane.SetYResolution(1)
            source = plane
        elif object_type == 'tetrahedron':
            source = vtk.vtkPlatonicSolidSource()
            source.SetSolidTypeToTetrahedron()
        elif object_type == 'octahedron':
            source = vtk.vtkPlatonicSolidSource()
            source.SetSolidTypeToOctahedron()
        elif object_type == 'icosahedron':
            source = vtk.vtkPlatonicSolidSource()
            source.SetSolidTypeToIcosahedron()
        elif object_type == 'dodecahedron':
            source = vtk.vtkPlatonicSolidSource()
            source.SetSolidTypeToDodecahedron()
        else:
            print(f"Unknown object type: {object_type}")
            return None, None

        self.sources.append(source)
        print(f"✓ {object_type.capitalize()} source created")
        mapper = self.create_mapper(source)
        actor = self.create_actor(mapper)
        self.current_object_name = object_type
        return actor, object_type
    
    def create_mapper(self, source):
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(source.GetOutputPort())

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(cleaner.GetOutputPort())
        normals.AutoOrientNormalsOn()
        normals.ConsistencyOn()
        normals.SplittingOn()
        normals.SetFeatureAngle(30.0)
        # Use point normals by default for Gouraud/Phong
        normals.ComputeCellNormalsOff()
        normals.ComputePointNormalsOn()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        mapper.InterpolateScalarsBeforeMappingOff()
        mapper.ScalarVisibilityOff()

        # Keep references so we can tweak normals later
        try:
            mapper._vt_cleaner = cleaner
            mapper._vt_normals = normals
        except Exception:
            pass

        self.mappers.append(mapper)
        return mapper
    
    def create_actor(self, mapper):
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        prop = actor.GetProperty()
        prop.LightingOn()
        prop.SetColor(self.current_color)
        prop.SetAmbient(0.4)  # CHANGED from 0.05 to 0.4 (much brighter in dark scenes)
        prop.SetDiffuse(1.0)
        prop.SetSpecular(0.30)
        prop.SetSpecularPower(40.0)
        prop.SetInterpolationToGouraud()
        prop.BackfaceCullingOff()
        prop.FrontfaceCullingOff()

        if hasattr(mapper, 'ScalarVisibilityOff'):
            mapper.ScalarVisibilityOff()

        self.actors.append(actor)
        return actor

    def setup_rendering_pipeline(self, vtk_widget):
        print("Setting up rendering pipeline...")
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.2, 0.2, 0.2)
        self.renderer.AutomaticLightCreationOff()
        self.renderer.SetTwoSidedLighting(True)

        self.window = vtk_widget.GetRenderWindow()
        self.window.AddRenderer(self.renderer)
        self.interactor = self.window.GetInteractor()

        # Make sure the interactor/context is initialized
        try:
            if hasattr(vtk_widget, "Initialize"):
                vtk_widget.Initialize()
        except Exception:
            pass

        self.setup_grid()
        self.setup_axis_lines()
        self.setup_axes_widget()
        self.setup_default_camera()
        print("✓ Rendering pipeline setup complete")

    def setup_grid(self):
        """Creates a Blender-like infinite grid with fading effect."""
        # Create a plane source for the grid, make it much larger for an "infinite" feel
        grid_size = 500
        grid_resolution = 1000 # Increase resolution for the larger size
        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(-grid_size, -grid_size, -0.01)
        plane.SetPoint1(grid_size, -grid_size, -0.01)
        plane.SetPoint2(-grid_size, grid_size, -0.01)
        plane.SetXResolution(grid_resolution)
        plane.SetYResolution(grid_resolution)
        plane.Update()
        
        grid_polydata = plane.GetOutput()
        
        # Create a mapper and resolve Z-fighting
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(grid_polydata)
        # This is the key to preventing the grid from rendering through objects
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-1, -1)

        # Create the grid actor
        self.grid_actor = vtk.vtkActor()
        self.grid_actor.SetMapper(mapper)
        self.grid_actor.GetProperty().SetRepresentationToWireframe()
        self.grid_actor.GetProperty().SetColor(0.294, 0.294, 0.294)
        self.grid_actor.GetProperty().SetOpacity(0.5)
        self.grid_actor.GetProperty().SetLineWidth(1)
        self.grid_actor.PickableOff()
        
        self.renderer.AddActor(self.grid_actor)

    def setup_axis_lines(self):
        """Creates colored axis lines (X=Red, Y=Green, Z=Blue)."""
        axis_length = 100
        axes_data = [
            # X-axis (Red)
            {'start': (0, 0, 0), 'end': (axis_length, 0, 0), 'color': (1, 0, 0)},
            {'start': (0, 0, 0), 'end': (-axis_length, 0, 0), 'color': (0.5, 0, 0)},
            # Y-axis (Green)
            {'start': (0, 0, 0), 'end': (0, axis_length, 0), 'color': (0, 1, 0)},
            {'start': (0, 0, 0), 'end': (0, -axis_length, 0), 'color': (0, 0.5, 0)},
            # Z-axis (Blue)
            {'start': (0, 0, 0), 'end': (0, 0, axis_length), 'color': (0, 0, 1)},
            {'start': (0, 0, 0), 'end': (0, 0, -axis_length), 'color': (0, 0, 0.5)},
        ]
        
        for axis_info in axes_data:
            # Create line source
            line_source = vtk.vtkLineSource()
            line_source.SetPoint1(axis_info['start'])
            line_source.SetPoint2(axis_info['end'])
            
            # Create mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(line_source.GetOutputPort())
            
            # Create actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(axis_info['color'])
            actor.GetProperty().SetLineWidth(2)
            actor.GetProperty().SetOpacity(0.8)
            
            # Make axis non-pickable and always on top
            actor.PickableOff()
            
            self.axis_actors.append(actor)
            self.renderer.AddActor(actor)

    def setup_default_camera(self):
        """Sets up the default camera view like Blender (perspective from top-right-front)."""
        camera = self.renderer.GetActiveCamera()
        
        # Blender's default camera position (7.36, -6.93, 4.96)
        # Adjusted for better viewing
        camera.SetPosition(20, -20, 15)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 0, 1)
        
        # Set perspective projection
        camera.SetParallelProjection(False)
        camera.SetViewAngle(40)  # Field of view
        
        self.renderer.ResetCameraClippingRange()

    def setup_axes_widget(self):
        """Creates and configures the orientation axes widget with clickable axis labels."""
        # Create the axes actor
        axes = vtk.vtkAxesActor()
        
        # Customize the axes appearance - MAKE IT BIGGER
        axes.SetTotalLength(7.5, 7.5, 7.5)
        axes.SetShaftTypeToLine()
        axes.SetAxisLabels(1)
        axes.SetCylinderRadius(0.08)
        
        # Make the labels larger
        axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(24)
        axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(24)
        axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(24)
        
        # Create the orientation marker widget
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(axes)
        self.axes_widget.SetInteractor(self.interactor)
        self.axes_widget.SetViewport(0.55, 0.55, 1.0, 1.0)  # Moved to top-right corner
        self.axes_widget.SetEnabled(1)
        self.axes_widget.InteractiveOff()  # Make it non-draggable/sticky
        
        # Track last clicked axis for flip functionality
        self.last_clicked_axis = None
        self.axis_flip_state = {'X': 1, 'Y': 1, 'Z': 1}  # 1 for positive, -1 for negative
        
        # Add custom interaction for clicking axes
        self.setup_axis_picker()

    def setup_axis_picker(self):
        """Sets up click interaction for the axis widget."""
        # Bump priority so we pre-empt the default camera style
        priority = 100.0
        if self.interactor and self.click_observer is None:
            self.click_observer = self.interactor.AddObserver('LeftButtonPressEvent', self.on_axis_click, priority)
        # Also swallow release to avoid camera spin when clicking the axes
        if self.interactor and self.axis_release_observer is None:
            self.axis_release_observer = self.interactor.AddObserver('LeftButtonReleaseEvent', self.on_axis_click, priority)
    
    def _suppress_camera_style_begin(self):
        try:
            if self.interactor and not self._axis_click_active:
                self._saved_style = self.interactor.GetInteractorStyle()
                # User style prevents TrackballCamera from rotating
                self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleUser())
                self._axis_click_active = True
        except Exception:
            pass

    def _suppress_camera_style_end(self):
        try:
            if self.interactor and self._axis_click_active:
                if self._saved_style:
                    self.interactor.SetInteractorStyle(self._saved_style)
                self._saved_style = None
                self._axis_click_active = False
        except Exception:
            pass
    
    def shutdown(self):
        """Stop rendering and detach VTK widgets/observers safely."""
        # Stop future renders
        self._alive = False
        # Disable interactive widgets
        try:
            if self.axes_widget:
                self.axes_widget.SetEnabled(0)
                self.axes_widget = None
        except Exception:
            pass
        try:
            if hasattr(self, "transform_widget") and self.transform_widget:
                self.transform_widget.Off()
                self.transform_widget = None
        except Exception:
            pass
        # Remove custom observer
        try:
            if self.interactor and self.click_observer is not None:
                self.interactor.RemoveObserver(self.click_observer)
                self.click_observer = None
            if self.interactor and self.axis_release_observer is not None:  # NEW
                self.interactor.RemoveObserver(self.axis_release_observer)
                self.axis_release_observer = None
        except Exception:
            pass
        # As an extra safety, avoid on-screen rendering during teardown
        try:
            if self.window:
                self.window.SetOffScreenRendering(1)
        except Exception:
            pass

    def on_axis_click(self, obj, event):
        """Handles clicks on the axis widget to change camera view."""
        click_pos = self.interactor.GetEventPosition()
        viewport = self.axes_widget.GetViewport()
        size = self.window.GetSize()
        x_min = int(viewport[0] * size[0]); y_min = int(viewport[1] * size[1])
        x_max = int(viewport[2] * size[0]); y_max = int(viewport[3] * size[1])

        if (x_min <= click_pos[0] <= x_max and y_min <= click_pos[1] <= y_max):
            # Swallow both press and release so TrackballCamera doesn't rotate
            try:
                obj.SetAbortFlag(1)
            except Exception:
                pass

            if event == "LeftButtonPressEvent":
                # Prevent TrackballCamera from entering rotate mode
                self._suppress_camera_style_begin()

            # IMPORTANT: Exit early on release
            if event == "LeftButtonReleaseEvent":
                # Restore camera style lock on mouse-up
                self._suppress_camera_style_end()
                return

            # Decide which axis was clicked (only on press)
            rel_x = (click_pos[0] - x_min) / max(1, (x_max - x_min))
            rel_y = (click_pos[1] - y_min) / max(1, (y_max - y_min))
            center_threshold = 0.15

            axis = None
            if rel_x > 0.6 and abs(rel_y - 0.5) < center_threshold:
                axis = 'X'
            elif rel_x < 0.4 and abs(rel_y - 0.5) < center_threshold:
                axis = 'X'
            elif abs(rel_x - 0.5) < center_threshold and rel_y > 0.6:
                axis = 'Z'
            elif abs(rel_x - 0.5) < center_threshold and rel_y < 0.4:
                axis = 'Z'
            elif abs(rel_x - 0.5) < center_threshold and abs(rel_y - 0.5) < center_threshold:
                axis = 'Y'

            if axis:
                if self.last_clicked_axis == axis:
                    self.axis_flip_state[axis] *= -1
                else:
                    self.axis_flip_state[axis] = 1
                    self.last_clicked_axis = axis
                self.animate_to_axis_view(axis, self.axis_flip_state[axis])
                self.render_all()
            return

    def animate_to_axis_view(self, axis, direction):
        """Smoothly animates camera to view along specified axis."""
        camera = self.renderer.GetActiveCamera()
        
        # Get current focal point (where camera is looking)
        focal_point = camera.GetFocalPoint()
        
        # Calculate distance from camera to focal point
        current_pos = camera.GetPosition()
        distance = ((current_pos[0] - focal_point[0])**2 + 
                   (current_pos[1] - focal_point[1])**2 + 
                   (current_pos[2] - focal_point[2])**2)**0.5
        
        # Define target positions and view-up vectors for each axis
        axis_views = {
            'X': {
                'position': (focal_point[0] + distance * direction, focal_point[1], focal_point[2]),
                'viewup': (0, 0, 1)  # Z-axis as "up"
            },
            'Y': {
                'position': (focal_point[0], focal_point[1] + distance * direction, focal_point[2]),
                'viewup': (0, 0, 1)  # Z-axis as "up"
            },
            'Z': {
                'position': (focal_point[0], focal_point[1], focal_point[2] + distance * direction),
                'viewup': (0, 1, 0) if direction > 0 else (0, -1, 0)  # Y-axis as "up"
            }
        }
        
        target_view = axis_views[axis]
        target_pos = target_view['position']
        target_up = target_view['viewup']
        
        # Animate camera movement
        self.animate_camera_transition(current_pos, target_pos, camera.GetViewUp(), target_up, focal_point)

    def animate_camera_transition(self, start_pos, end_pos, start_up, end_up, focal_point, steps=30):
        """Smoothly interpolates camera from start to end position with lifecycle guards."""
        camera = self.renderer.GetActiveCamera()
        if not camera or not self.renderer or not self.window:
            return
    
        try:
            steps = max(1, int(steps))
        except Exception:
            steps = 30
    
        # Block input during animation
        try:
            if self.interactor:
                self.interactor.Disable()
        except Exception:
            pass
    
        fx, fy, fz = focal_point
    
        for i in range(steps + 1):
            # Stop if window is gone or not drawable (prevents wglMakeCurrent errors on Windows)
            if not self._alive:
                break
            try:
                if hasattr(self.window, "IsDrawable") and not self.window.IsDrawable():
                    break
                if hasattr(self.window, "GetMapped") and not self.window.GetMapped():
                    break
            except Exception:
                break
    
            # Smoothstep easing
            t = i / float(steps)
            t = t * t * (3.0 - 2.0 * t)
    
            # Interpolate position and view-up
            pos = (
                start_pos[0] + (end_pos[0] - start_pos[0]) * t,
                start_pos[1] + (end_pos[1] - start_pos[1]) * t,
                start_pos[2] + (end_pos[2] - start_pos[2]) * t,
            )
            up = (
                start_up[0] + (end_up[0] - start_up[0]) * t,
                start_up[1] + (end_up[1] - start_up[1]) * t,
                start_up[2] + (end_up[2] - start_up[2]) * t,
            )
    
            camera.SetPosition(pos)
            camera.SetFocalPoint(fx, fy, fz)
            camera.SetViewUp(up)
    
            try:
                self.renderer.ResetCameraClippingRange()
            except Exception:
                pass
            self.render_all()
    
            # Keep UI responsive and pace the animation
            try:
                QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.AllEvents, 1)
                QtCore.QThread.msleep(10)
            except Exception:
                pass
    
        # Re-enable input
        try:
            if self.interactor:
                self.interactor.Enable()
        except Exception:
            pass
    
        # Safety: ensure interactor style is restored even if mouse release was missed
        try:
            self._suppress_camera_style_end()
        except Exception:
            pass

    def add_actor_to_scene(self, actor):
        if actor:
            self.renderer.AddActor(actor)
            # Track all scene actors (meshes, gizmos) for proper cleanup
            if actor not in self.actors:
                self.actors.append(actor)
            print(f"✓ Added actor to scene")
        # Use ResetCameraClippingRange instead of ResetCamera to avoid zooming out
        self.renderer.ResetCameraClippingRange()
        self.render_all()

    def clear_scene(self):
        print("Clearing scene...")
        # Don't remove grid and axis actors
        for actor in self.actors:
            self.renderer.RemoveActor(actor)
        self.actors.clear()
        self.mappers.clear()
        self.sources.clear()
        # Remove user-added lights
        for l in self.lights:
            self.renderer.RemoveLight(l)
        self.lights.clear()
        self.render_all()

    def render_all(self):
        # Guard against rendering during teardown or when not drawable
        if not self._alive or not self.window:
            return
        try:
            if hasattr(self.window, "IsDrawable") and not self.window.IsDrawable():
                return
            if hasattr(self.window, "GetMapped") and not self.window.GetMapped():
                return
            self.window.Render()
        except Exception:
            # Swallow render errors during shutdown on Windows
            pass

    def change_color(self, color, actor=None):
        if actor:
            actor.GetProperty().SetColor(color)
            self.render_all()

    def start(self, vtk_widget):
        print("=" * 60)
        print("Starting Interactive VTK Application")
        print("=" * 60)
        self.setup_rendering_pipeline(vtk_widget)
        self.render_all()
        print("✓ VTK application started successfully")
        print("=" * 60)

    # NEW: light helpers
    def create_light(self, light_type: str):
        """Create a vtkLight configured as Point/Directional/Spot with clearer defaults."""
        lt = light_type.lower()
        light = vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetSwitch(True)
        # Higher intensity so differences show
        light.SetIntensity(1.5 if lt == "directional" else 1.0)
        light.SetColor(1.0, 1.0, 1.0)
        light.SetPosition(10, -10, 10)
        light.SetFocalPoint(0, 0, 0)

        if lt == "point":
            light.SetPositional(True)
            light.SetConeAngle(180.0)
            light.SetExponent(1.0)
        elif lt == "directional":
            light.SetPositional(False)  # Infinite, parallel rays
        elif lt == "spot":
            light.SetPositional(True)
            light.SetConeAngle(25.0)   # narrower cone for visible falloff
            light.SetExponent(15.0)    # sharper penumbra
        else:
            print(f"Unknown light type: {light_type}")
            return None
        return light
    
    def add_light_to_scene(self, light: vtk.vtkLight):
        """Add a light to the renderer and track it."""
        if not light:
            return
        self.renderer.AddLight(light)
        self.lights.append(light)
        self.render_all()

    def remove_light_from_scene(self, light: vtk.vtkLight):
        if not light:
            return
        self.renderer.RemoveLight(light)
        if light in self.lights:
            self.lights.remove(light)
        self.render_all()

class CameraPropertiesDialog(QtWidgets.QDialog):
    def __init__(self, renderer, parent=None):
        super().__init__(parent)
        self.renderer = renderer
        self.setWindowTitle("Camera Properties")
        layout = QtWidgets.QFormLayout(self)
        self.spinboxes = {}
        camera = self.renderer.GetActiveCamera()
        pos, focal, up = camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp()
        for prop_name, values in [("Position", pos), ("Focal Point", focal), ("View Up", up)]:
            for i, axis in enumerate("XYZ"):
                key = f"{prop_name}{axis}"
                self.spinboxes[key] = QtWidgets.QDoubleSpinBox()
                self.spinboxes[key].setRange(-1000, 1000)
                self.spinboxes[key].setValue(values[i])
                layout.addRow(f"{prop_name} {axis}", self.spinboxes[key])
        self.projection_combo = QtWidgets.QComboBox()
        self.projection_combo.addItems(["Perspective", "Parallel"])
        self.projection_combo.setCurrentIndex(0 if camera.GetParallelProjection() == 0 else 1)
        layout.addRow("Projection", self.projection_combo)
        apply_button = QtWidgets.QPushButton("Apply")
        apply_button.clicked.connect(self.apply_changes)
        layout.addRow(apply_button)

    def apply_changes(self):
        camera = self.renderer.GetActiveCamera()
        pos = [self.spinboxes[f"Position{ax}"].value() for ax in "XYZ"]
        focal = [self.spinboxes[f"Focal Point{ax}"].value() for ax in "XYZ"]
        up = [self.spinboxes[f"View Up{ax}"].value() for ax in "XYZ"]
        camera.SetPosition(pos)
        camera.SetFocalPoint(focal)
        camera.SetViewUp(up)
        camera.SetParallelProjection(self.projection_combo.currentIndex() == 1)
        self.renderer.ResetCameraClippingRange()
        self.parent().vtk_app.render_all()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VTK 3D Editor")
        self.resize(1600, 900)
        self.vtkWidget = QVTKRenderWindowInteractor()
        self.setCentralWidget(self.vtkWidget)
        try:
            self.vtkWidget.Initialize()
        except Exception:
            pass
        self.vtk_app = myVTK()
        self.vtk_app.start(self.vtkWidget)
        self.undo_stack = QtWidgets.QUndoStack(self)
        self._pending_transform = None
        self._pending_prop_snapshot = None
        self.object_registry = {}
        self.light_registry = {}
        self.transform_widget = None
        self.current_transform_mode = 'translate'
        self.block_signals = False
        self.current_selected_actor = None
        self.current_tool = None
        self.snap_increment = 0.5
        self.clipboard = None
        self.actor_texture_paths = {}
        # Camera mode state
        self.camera_mode = False
        self._prev_interactor_style = None
        self._camera_obs = []
        self.camera_mode_label = None
        self.current_object_name = 'sphere'
        self._scale_uniform_lock = False
        self._scale_obs = []                # interactor observers for scale mode
        self._uniform_hint_actors = []      # two billboard text actors "⇔"

        # Build UI and scene with the ORIGINAL Qt look
        self.create_ui()
        self.create_initial_scene()

        # 🔹 Save the original Qt palette / stylesheet / VTK background
        app = QtWidgets.QApplication.instance()
        self.original_palette = app.palette()
        self.original_stylesheet = app.styleSheet()
        self.original_bg = self.vtk_app.renderer.GetBackground()

        # 🔹 Now apply Blender theme as the default startup theme
        self.apply_theme("blender")

        self.show()

    def create_theme_actions(self):
        
        """Create theme selection actions with Blender-like option."""
        self.theme_menu = QtWidgets.QMenu("Theme", self)

        self.blender_theme_action = QtWidgets.QAction("Blender Theme", self, checkable=True)
        self.blender_theme_action.triggered.connect(lambda: self.apply_theme("blender"))

        self.light_theme_action = QtWidgets.QAction("Light Theme", self, checkable=True)
        self.light_theme_action.triggered.connect(lambda: self.apply_theme("light"))

        # Create an action group to make themes mutually exclusive
        self.theme_group = QtWidgets.QActionGroup(self)
        self.theme_group.addAction(self.blender_theme_action)
        self.theme_group.addAction(self.light_theme_action)

        # Set Blender theme as default
        self.blender_theme_action.setChecked(True)

        self.theme_menu.addAction(self.blender_theme_action)
        self.theme_menu.addAction(self.light_theme_action)

    def apply_theme(self, theme_name):
        """Apply the selected theme to the application."""
        app = QtWidgets.QApplication.instance()

        # Clear previous stylesheet and set a neutral style each time
        app.setStyleSheet("")
        app.setStyle(QStyleFactory.create("Fusion"))

        base_palette = app.style().standardPalette()
        app.setPalette(base_palette)

        palette = QPalette(base_palette)
        self.vtk_app.renderer.SetBackground(*self.original_bg)

        if theme_name == "blender":
            # Blender-inspired palette
            palette.setColor(QPalette.Window, QColor(50, 50, 50))
            palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
            palette.setColor(QPalette.Base, QColor(60, 60, 60))
            palette.setColor(QPalette.AlternateBase, QColor(55, 55, 55))
            palette.setColor(QPalette.Text, QColor(220, 220, 220))
            palette.setColor(QPalette.BrightText, QColor(255, 255, 255))
            palette.setColor(QPalette.Button, QColor(72, 72, 72))
            palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
            palette.setColor(QPalette.Highlight, QColor(242, 142, 55))
            palette.setColor(QPalette.HighlightedText, QColor(20, 20, 20))
            palette.setColor(QPalette.Link, QColor(93, 175, 255))
            palette.setColor(QPalette.ToolTipBase, QColor(45, 45, 45))
            palette.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
            palette.setColor(QPalette.Disabled, QPalette.Text, QColor(120, 120, 120))
            palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(120, 120, 120))
            palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(120, 120, 120))

            self.vtk_app.renderer.SetBackground(0.235, 0.235, 0.235)

            css = """
            QMainWindow { background-color: #323232; }
            QMenuBar { background-color: #3c3c3c; color: #dcdcdc; border-bottom: 1px solid #1a1a1a; }
            QMenuBar::item:selected { background-color: #f28e37; color: #141414; }
            QMenu { background-color: #3c3c3c; color: #dcdcdc; border: 1px solid #1a1a1a; }
            QMenu::item:selected { background-color: #f28e37; color: #141414; }
            QToolBar { background-color: #3c3c3c; border: none; spacing: 3px; padding: 4px; }
            QToolButton { background-color: #484848; border: 1px solid #2a2a2a; border-radius: 2px; padding: 4px; color: #dcdcdc; }
            QToolButton:hover { background-color: #5a5a5a; }
            QToolButton:pressed { background-color: #3a3a3a; }
            QToolButton:checked { background-color: #f28e37; color: #141414; border: 1px solid #f28e37; }
            QPushButton { background-color: #484848; border: 1px solid #2a2a2a; border-radius: 2px; padding: 4px 8px; color: #dcdcdc; }
            QPushButton:hover { background-color: #5a5a5a; }
            QPushButton:pressed { background-color: #3a3a3a; }
            QPushButton:checked { background-color: #f28e37; color: #141414; border: 1px solid #f28e37; }
            /* remaining existing dark theme rules ... */
            """
            self.vtk_app.renderer.SetBackground(0.235, 0.235, 0.235)

        elif theme_name == "light":
            palette.setColor(QPalette.Window, QColor(240, 240, 240))
            palette.setColor(QPalette.Base, QColor(255, 255, 255))
            palette.setColor(QPalette.Text, QColor(0, 0, 0))
            palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
            palette.setColor(QPalette.Button, QColor(240, 240, 240))
            palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
            palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
            palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

            self.vtk_app.renderer.SetBackground(0.85, 0.85, 0.85)

            # Explicit QToolButton rules prevent fallback to dark theme on hover
            css = """
            QMainWindow { background-color: #f0f0f0; }
            QMenuBar, QToolBar { background-color: #f7f7f7; color: #000000; }
            QStatusBar { background-color: #f7f7f7; color: #000000; }
            QToolBar { border: 0px; padding: 4px; spacing: 6px; }
            QToolButton {
                background: #f2f2f2;
                color: #000000;
                border: 1px solid #c6c6c6;
                border-radius: 3px;
                padding: 4px;
            }
            QToolButton:hover {
                background: #e4e4e4;
                border: 1px solid #b5b5b5;
            }
            QToolButton:pressed {
                background: #d8d8d8;
            }
            QToolButton:checked {
                background: #0078d7;
                color: #ffffff;
                border: 1px solid #0078d7;
            }
            QPushButton {
                background: #eaeaea;
                border: 1px solid #bcbcbc;
                border-radius: 3px;
                padding: 4px 8px;
                color: #000000;
            }
            QPushButton:hover { background: #dedede; }
            QPushButton:pressed { background: #d2d2d2; }
            QDockWidget { background-color: #f5f5f5; color: #000000; }
            QDockWidget::title { background-color: #e5e5e5; color: #000000; padding: 4px; }
            QTreeWidget {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #cccccc;
                selection-background-color: #0078d7;
                selection-color: #ffffff;
            }
            QTabWidget::pane { border: 1px solid #c8c8c8; background: #ffffff; }
            QTabBar::tab {
                background: #e0e0e0;
                color: #000000;
                padding: 6px 12px;
                border: 1px solid #c0c0c0;
                border-bottom: 2px solid #c0c0c0;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                border-bottom: 2px solid #0078d7;
            }
            QTabBar::tab:hover {
                background: #ececec;
            }
            QScrollBar:vertical { background: #f0f0f0; width: 12px; border: none; }
            QScrollBar::handle:vertical { background: #c8c8c8; min-height: 20px; border-radius: 6px; }
            QScrollBar::handle:vertical:hover { background: #b5b5b5; }
            QScrollBar:horizontal { background: #f0f0f0; height: 12px; border: none; }
            QScrollBar::handle:horizontal { background: #c8c8c8; min-width: 20px; border-radius: 6px; }
            QScrollBar::handle:horizontal:hover { background: #b5b5b5; }
            QSlider::groove:horizontal { border: 1px solid #c6c6c6; height: 4px; background: #dcdcdc; margin: 2px 0; }
            QSlider::handle:horizontal {
                background: #0078d7;
                border: 1px solid #006bbf;
                width: 12px;
                margin: -4px 0;
                border-radius: 3px;
            }
            QCheckBox::indicator {
                width: 14px; height: 14px;
                border: 1px solid #b5b5b5;
                border-radius: 3px;
                background: #ffffff;
            }
            QCheckBox::indicator:hover { border: 1px solid #0078d7; }
            QCheckBox::indicator:checked {
                background: #0078d7;
                border: 1px solid #0078d7;
            }
            """
        else:
            css = ""

        app.setPalette(palette)
        if css:
            app.setStyleSheet(css)

        self._repolish_ui()
        self.vtk_app.render_all()
        self.statusBar().showMessage(f"Theme changed to: {theme_name.capitalize()}")

    def create_ui(self):
        self.create_actions()
        self.create_theme_actions()
        self.create_menu_bar()
        self.create_tool_bar()
        self.create_status_bar()
        self.create_dock_widgets()
        self.create_transform_mode_buttons() 

    def create_initial_scene(self):
        self.add_new_object('sphere', self.vtk_app.create_object)
        
        # Add a strong default light to illuminate loaded models
        self.add_new_light("Point")
        # Move it to a good position
        if "point_light_1" in self.light_registry:
            light = self.light_registry["point_light_1"]["light"]
            light.SetPosition(50, 50, 100)
            light.SetIntensity(3.0)

    def create_actions(self):
        self.open_file_action = QtWidgets.QAction("Open Model...", self, triggered=self.on_open_file)
        self.exit_action = QtWidgets.QAction("E&xit", self, shortcut="Ctrl+Q", triggered=self.close)
        self.new_scene_action = QtWidgets.QAction("New Scene Window", self, shortcut="Ctrl+N", triggered=self.on_new_scene)
    
        # Icons via central ICON_DIR (see ICON_DIR and _icon at top)
        self.create_sphere_action = QtWidgets.QAction(
            _icon('sphere_icon.png', self.style(), QtWidgets.QStyle.SP_FileIcon),
            "Sphere", self, triggered=lambda: self.add_new_object('sphere', self.vtk_app.create_object))
    
        self.create_cube_action = QtWidgets.QAction(
            _icon('cube_icon.png', self.style(), QtWidgets.QStyle.SP_FileIcon),
            "Cube", self, triggered=lambda: self.add_new_object('cube', self.vtk_app.create_object))
    
        self.add_cube_tool_action = QtWidgets.QAction(
            _icon('add_cube_tool_icon.png', self.style(), QtWidgets.QStyle.SP_DialogOpenButton),
            "Add Cube (Tool)", self, triggered=self.activate_add_cube_tool)
    
        self.create_subdiv_cube_action = QtWidgets.QAction(
            "Subdivided Cube...", self, triggered=self.on_create_subdivided_cube)
    
        self.create_tetrahedron_action = QtWidgets.QAction("Tetrahedron", self,
            triggered=lambda: self.add_new_object('tetrahedron', self.vtk_app.create_object))
        self.create_octahedron_action = QtWidgets.QAction("Octahedron", self,
            triggered=lambda: self.add_new_object('octahedron', self.vtk_app.create_object))
        self.create_icosahedron_action = QtWidgets.QAction("Icosahedron", self,
            triggered=lambda: self.add_new_object('icosahedron', self.vtk_app.create_object))
        self.create_dodecahedron_action = QtWidgets.QAction("Dodecahedron", self,
            triggered=lambda: self.add_new_object('dodecahedron', self.vtk_app.create_object))
    
        self.create_reduced_cube_action = QtWidgets.QAction("Reduced Cube", self,
            triggered=lambda: self.add_new_object('reduced_cube', self.vtk_app.create_reduced_cube))
    
        self.create_cone_action = QtWidgets.QAction("Cone", self,
            triggered=lambda: self.add_new_object('cone', self.vtk_app.create_object))
        self.create_cylinder_action = QtWidgets.QAction("Cylinder", self,
            triggered=lambda: self.add_new_object('cylinder', self.vtk_app.create_object))
        self.create_pyramid_action = QtWidgets.QAction("Pyramid", self,
            triggered=lambda: self.add_new_object('pyramid', self.vtk_app.create_object))
        self.create_rectangle_action = QtWidgets.QAction("Rectangle", self,
            triggered=lambda: self.add_new_object('rectangle', self.vtk_app.create_object))

        self.create_param_torus_action = QtWidgets.QAction("Parametric Torus", self,
            triggered=lambda: self.add_new_object('torus', self.vtk_app.create_parametric))
        self.create_param_klein_action = QtWidgets.QAction("Klein Surface", self,
            triggered=lambda: self.add_new_object('klein', self.vtk_app.create_parametric))
    
        self.create_quadric_action = QtWidgets.QAction("Quadric Sphere", self,
            triggered=lambda: self.add_new_object('quadric_sphere', self.vtk_app.create_implicit_object))
        self.create_torus_action = QtWidgets.QAction("Torus", self,
            triggered=lambda: self.add_new_object('torus', self.vtk_app.create_implicit_object))
    
        self.create_convex_point_set_action = QtWidgets.QAction("Convex Point Set", self,
            triggered=lambda: self.add_new_object('convex_point_set', self.vtk_app.create_cell_object))
        self.create_polyhedron_cell_action = QtWidgets.QAction("Polyhedron Cell", self,
            triggered=lambda: self.add_new_object('polyhedron_cell', self.vtk_app.create_cell_object))
    
        self.clear_scene_action = QtWidgets.QAction(
            _icon('reset_icon.png', self.style(), QtWidgets.QStyle.SP_BrowserStop),
            "Clear Scene", self, triggered=self.clear_scene)
    
        self.reset_camera_action = QtWidgets.QAction("Reset Camera", self,
            triggered=lambda: self.vtk_app.renderer.ResetCamera())
        self.camera_props_action = QtWidgets.QAction("Camera Properties...", self,
            triggered=self.open_camera_dialog)
    
        self.toggle_grid_action = QtWidgets.QAction("Show Grid", self, checkable=True,
            triggered=self.toggle_grid_visibility)
        self.toggle_grid_action.setChecked(True)
    
        self.toggle_gizmo_action = QtWidgets.QAction("Show Transform Gizmo", self, checkable=True,
            triggered=self.toggle_gizmo)
        self.toggle_gizmo_action.setChecked(True)
    
        self.create_point_light_action = QtWidgets.QAction("Point Light", self,
            triggered=lambda: self.add_new_light("Point"))
        self.create_directional_light_action = QtWidgets.QAction("Directional Light", self,
            triggered=lambda: self.add_new_light("Directional"))
        self.create_spot_light_action = QtWidgets.QAction("Spot Light", self,
            triggered=lambda: self.add_new_light("Spot"))
    
        self.toggle_lighting_action = QtWidgets.QAction("Lighting On", self, checkable=True,
            triggered=self.on_toggle_lighting)
        self.toggle_lighting_action.setChecked(True)
        self.lighting_enabled = True
    
        self.export_selected_action = QtWidgets.QAction("Export Selected...", self,
            triggered=self.export_selected)
        self.export_all_action = QtWidgets.QAction("Export All To Directory...", self,
            triggered=self.export_all_to_directory)
        self.export_scene_action = QtWidgets.QAction("Export Scene As One Mesh...", self,
            triggered=self.export_scene_as_one)
        self.export_scene_multi_obj_action = QtWidgets.QAction("Export Scene (Multi-Object OBJ)...", self,
            triggered=self.export_scene_multi_obj)
    
        self.copy_action = QtWidgets.QAction("Copy", self, shortcut="Ctrl+C",
            triggered=self.on_copy_selected)
        self.paste_action = QtWidgets.QAction("Paste", self, shortcut="Ctrl+V",
            triggered=self.on_paste_selected)
        self.addAction(self.copy_action)
        self.addAction(self.paste_action)
    
        self.vertex_edit_action = QtWidgets.QAction("Vertex Edit Tool", self,
            checkable=True, triggered=self.on_vertex_edit_toggled)
        
        CUSTOM_CAMERA_ICON = 'camera_mode8.png'  # put this file in ICON_DIR
        self.camera_mode_action = QtWidgets.QAction(
            _icon(CUSTOM_CAMERA_ICON, self.style(), QtWidgets.QStyle.SP_DesktopIcon),
            "Camera Mode", self, checkable=True
        )
        self.camera_mode_action.triggered.connect(self.on_camera_mode_toggled)
    
        self.add_cube_tool_action.setShortcut("")  # optional: no shortcut
    
        self.undo_action = self.undo_stack.createUndoAction(self, "Undo")
        self.undo_action.setShortcut("Ctrl+Z")
        self.redo_action = self.undo_stack.createRedoAction(self, "Redo")
        self.redo_action.setShortcut("Ctrl+Y")

    def create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.open_file_action)
        file_menu.addAction(self.new_scene_action)
        file_menu.addSeparator()
        file_menu.addAction(self.open_file_action)
        file_menu.addAction(self.clear_scene_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_selected_action)
        file_menu.addAction(self.export_all_action)
        file_menu.addAction(self.export_scene_action)
        file_menu.addAction(self.export_scene_multi_obj_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action) 

        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction(self.undo_action)   
        edit_menu.addAction(self.redo_action)  
        edit_menu.addSeparator()
        edit_menu.addAction(self.copy_action)
        edit_menu.addAction(self.paste_action)
        
        create_menu = menubar.addMenu("&Create")
        primitives_menu = create_menu.addMenu("Primitives")
        primitives_menu.addAction(self.create_sphere_action)
        primitives_menu.addAction(self.create_cube_action)
        primitives_menu.addAction(self.create_subdiv_cube_action)   
        primitives_menu.addAction(self.create_reduced_cube_action)
        primitives_menu.addSeparator()
        primitives_menu.addAction(self.create_cone_action)
        primitives_menu.addAction(self.create_cylinder_action)
        primitives_menu.addAction(self.create_pyramid_action)
        primitives_menu.addAction(self.create_rectangle_action)

        param_menu = create_menu.addMenu("Parametric")
        param_menu.addAction(self.create_param_torus_action)
        param_menu.addAction(self.create_param_klein_action)
        
        # NEW: tool
        create_menu.addAction(self.add_cube_tool_action)
        
        # Add Platonic solids submenu
        platonic_menu = create_menu.addMenu("Platonic Solids")
        platonic_menu.addAction(self.create_tetrahedron_action)
        platonic_menu.addAction(self.create_octahedron_action)
        platonic_menu.addAction(self.create_icosahedron_action)
        platonic_menu.addAction(self.create_dodecahedron_action)

        # NEW: Implicit surfaces submenu
        implicit_menu = create_menu.addMenu("Implicit Surfaces")
        implicit_menu.addAction(self.create_quadric_action)
        implicit_menu.addAction(self.create_torus_action)
        
        cell_menu = create_menu.addMenu("Cell Formats")
        cell_menu.addAction(self.create_convex_point_set_action)
        cell_menu.addAction(self.create_polyhedron_cell_action)  

        # NEW: Lights submenu
        lights_menu = create_menu.addMenu("Lights")
        lights_menu.addAction(self.create_point_light_action)
        lights_menu.addAction(self.create_directional_light_action)
        lights_menu.addAction(self.create_spot_light_action)

        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.toggle_grid_action)
        view_menu.addAction(self.toggle_lighting_action)  # NEW
        view_menu.addSeparator()
        view_menu.addAction(self.reset_camera_action)
        view_menu.addAction(self.camera_props_action)
        
        transform_menu = menubar.addMenu("&Transform")
        transform_menu.addAction(self.toggle_gizmo_action)
        transform_menu.addSeparator()
        #transform_menu.addAction(self.vertex_edit_action)
        
        menubar.addMenu(self.theme_menu)

    def export_scene_multi_obj(self):
        """Export all mesh actors into one OBJ file preserving per-object blocks."""
        if not self.object_registry:
            QtWidgets.QMessageBox.information(self, "Export Multi-Object OBJ", "No mesh objects in scene.")
            return
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Scene (Multi-Object OBJ)",
            "scene_multi.obj",
            "Wavefront OBJ (*.obj)"
        )
        if not filepath:
            return

        # Build OBJ text
        lines = []
        v_offset = 0

        for name, actor in self.object_registry.items():
            poly = self.polydata_from_actor(actor, apply_transform=True)
            if not poly or poly.GetNumberOfPoints() == 0:
                continue

            lines.append(f"o {name}")
            pts = poly.GetPoints()
            for i in range(pts.GetNumberOfPoints()):
                x, y, z = pts.GetPoint(i)
                lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")

            # Preserve original polygon topology (avoid forced triangulation to prevent diagonal seams)
            polys = poly.GetPolys()
            polys.InitTraversal()
            id_list = vtk.vtkIdList()
            while polys.GetNextCell(id_list):
                # OBJ indices are 1-based; add v_offset
                if id_list.GetNumberOfIds() >= 3:
                    face_idx = " ".join(str(v_offset + id_list.GetId(j) + 1) for j in range(id_list.GetNumberOfIds()))
                    lines.append(f"f {face_idx}")

            v_offset += pts.GetNumberOfPoints()

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Multi-Object OBJ", f"Failed to write file:\n{e}")
            return

        self.statusBar().showMessage(f"Exported {len(self.object_registry)} objects to {os.path.basename(filepath)} (multi-object OBJ)")

    def debug_actor(self, actor):
        if not actor:
            print("Debug: actor is None")
            return
        b = actor.GetBounds()
        print(f"Debug bounds: {b}")
        print(f"Debug position: {actor.GetPosition()}")
        print(f"Debug scale: {actor.GetScale()}")
        print(f"Debug opacity: {actor.GetProperty().GetOpacity()}")
        print(f"Debug visibility: {actor.GetVisibility()}")

    def on_open_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Model File",
            "",
            "3D Models (*.obj *.stl *.ply *.vtk *.vtp *.3ds);;All Files (*)"
        )
        if not file_path:
            return

        ext = os.path.splitext(file_path)[1].lower()
        base = os.path.splitext(os.path.basename(file_path))[0]
        collection = self.ensure_collection(base)

        if ext == ".3ds":
            actors = self.vtk_app.load_3ds_scene(file_path)
            if not actors:
                QtWidgets.QMessageBox.warning(self, "3DS Import", "Failed or empty 3DS scene.")
                return
            for actor, name in actors:
                if not actor:
                    continue
                self.vtk_app.orient_actor_y_up_to_z_up(actor)
                b = actor.GetBounds()
                if b and (abs(b[1]-b[0]) < 0.01 and abs(b[3]-b[2]) < 0.01 and abs(b[5]-b[4]) < 0.01):
                    actor.SetScale(100.0, 100.0, 100.0)

                count = len([k for k in self.object_registry if k.startswith(name)]) + 1
                unique = f"{name}_{count}" if count > 1 else name
                self.object_registry[unique] = actor
                actor.SetUserTransform(None)
                self.vtk_app.add_actor_to_scene(actor)
                self._make_object_item(unique, "mesh", collection)
                print(f"[3DS] Added part: {unique}")

            if collection.childCount() > 0:
                self.scene_outliner.setCurrentItem(collection.child(collection.childCount()-1))
                self.on_outliner_selection_changed(self.scene_outliner.currentItem())
            collection.setExpanded(True)
            self.update_scene_totals()
            self.statusBar().showMessage(f"Imported 3DS scene: {os.path.basename(file_path)}")
            return

        if ext == ".obj":
            parts = self._split_obj_to_temp_parts(file_path)
            if not parts:
                actor, name = self.vtk_app.load_file(file_path)
                if actor:
                    count = len([k for k in self.object_registry if k.startswith(name)]) + 1
                    unique = f"{name}_{count}" if count > 1 else name
                    self.object_registry[unique] = actor
                    actor.SetUserTransform(None)
                    self.vtk_app.add_actor_to_scene(actor)
                    self._make_object_item(unique, "mesh", collection)
                    self.scene_outliner.setCurrentItem(collection.child(collection.childCount()-1))
                    self.on_outliner_selection_changed(self.scene_outliner.currentItem())
                    collection.setExpanded(True)
                    self.update_scene_totals()
                return
            for temp_path, obj_name in parts:
                actor, _ = self.vtk_app.load_file(temp_path)
                if not actor:
                    continue
                count = len([k for k in self.object_registry if k.startswith(obj_name)]) + 1
                unique = f"{obj_name}_{count}" if count > 1 else obj_name
                self.object_registry[unique] = actor
                actor.SetUserTransform(None)
                self.vtk_app.add_actor_to_scene(actor)
                self._make_object_item(unique, "mesh", collection)

            if collection.childCount() > 0:
                self.scene_outliner.setCurrentItem(collection.child(0))
                self.on_outliner_selection_changed(collection.child(0))
            collection.setExpanded(True)
            self.update_scene_totals()
            self.statusBar().showMessage(f"Imported OBJ as collection '{base}' ({collection.childCount()} objects)")
            return

        actor, name = self.vtk_app.load_file(file_path)
        if actor:
            count = len([k for k in self.object_registry if k.startswith(name)]) + 1
            unique = f"{name}_{count}" if count > 1 else name
            self.object_registry[unique] = actor
            actor.SetUserTransform(None)
            self.vtk_app.add_actor_to_scene(actor)
            self._make_object_item(unique, "mesh", collection)
            self.scene_outliner.setCurrentItem(collection.child(collection.childCount()-1))
            self.on_outliner_selection_changed(self.scene_outliner.currentItem())
            collection.setExpanded(True)
            self.update_scene_totals()
            self.statusBar().showMessage(f"Imported {name}")

    def _split_obj_to_temp_parts(self, file_path: str):
        """Split an OBJ by 'o <name>' blocks into temp files. Returns [(temp_path, name)]."""
        parts = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception:
            return parts
    
        # Gather global tables to prepend to each chunk
        v = []; vt = []; vn = []; mtllibs = []; cur_faces = []; cur_name = None
    
        for ln in lines:
            if ln.startswith("mtllib "):
                mtllibs.append(ln)
            elif ln.startswith("v "):
                v.append(ln)
            elif ln.startswith("vt "):
                vt.append(ln)
            elif ln.startswith("vn "):
                vn.append(ln)
            elif ln.startswith("o "):
                # Commit previous object
                if cur_name and cur_faces:
                    tmp = self._emit_obj_chunk(file_path, cur_name, mtllibs, v, vt, vn, cur_faces)
                    parts.append(tmp)
                cur_name = ln.strip()[2:].strip() or "Object"
                cur_faces = []
            else:
                # Collect group, usemtl, s, f, etc. into current
                if cur_name is None:
                    # start default
                    cur_name = "Object"
                if ln.startswith(("g ", "usemtl ", "s ", "f ", "l ", "p ")):
                    cur_faces.append(ln)
    
        if cur_name and cur_faces:
            tmp = self._emit_obj_chunk(file_path, cur_name, mtllibs, v, vt, vn, cur_faces)
            parts.append(tmp)
    
        return parts
    
    def _emit_obj_chunk(self, src_path, name, mtllibs, v, vt, vn, faces):
        """Write one temp OBJ part, duplicating vertex tables (no reindexing)."""
        import tempfile
        dir_ = os.path.dirname(src_path)
        fd, tmp_path = tempfile.mkstemp(prefix=f"{name}_", suffix=".obj", dir=dir_)
        os.close(fd)
        with open(tmp_path, "w", encoding="utf-8") as out:
            for m in mtllibs:
                out.write(m)
            out.write(f"o {name}\n")
            for lst in (v, vt, vn):
                out.writelines(lst)
            out.writelines(faces)
        return (tmp_path, name)


    # NEW: open a fresh scene in a new top-level window and keep a reference on the QApplication
    def on_new_scene(self):
        app = QtWidgets.QApplication.instance()
        if not hasattr(app, "_scene_windows"):
            app._scene_windows = []
        w = MainWindow(parent=None)
        app._scene_windows.append(w)
        w.show()

    def toggle_grid_visibility(self, checked):
        """Toggles the visibility of the background grid."""
        if self.vtk_app.grid_actor:
            self.vtk_app.grid_actor.SetVisibility(checked)
            self.vtk_app.render_all()
            status = "shown" if checked else "hidden"
            self.statusBar().showMessage(f"Grid is now {status}")

    def create_tool_bar(self):
        toolbar = self.addToolBar("Main Toolbar")
        toolbar.addAction(self.open_file_action)
        toolbar.addSeparator()
        toolbar.addAction(self.create_sphere_action)
        toolbar.addAction(self.create_cube_action)
        #toolbar.addAction(self.create_subdiv_cube_action)  # NEW

        # NEW: quick Add Cube tool button
        toolbar.addAction(self.add_cube_tool_action)
        toolbar.addSeparator()
        # NEW: quick light buttons
        toolbar.addAction(self.create_point_light_action)
        toolbar.addAction(self.create_directional_light_action)
        toolbar.addAction(self.create_spot_light_action)
        toolbar.addSeparator()
        toolbar.addAction(self.toggle_lighting_action)  # NEW
        toolbar.addSeparator()
        #toolbar.addAction(self.vertex_edit_action)
        toolbar.addAction(self.clear_scene_action)
        toolbar.addSeparator()
        toolbar.addAction(self.camera_mode_action)

    def create_status_bar(self):
        self.statusBar().showMessage("Ready")

    def create_dock_widgets(self):
        # LEFT DOCK: Scene Collection (QTreeWidget)
        self.left_dock = QtWidgets.QDockWidget("Scene", self)
        self.left_dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea)
        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(6)
    
        outliner_group = QtWidgets.QGroupBox("Scene Collection")
        outliner_group.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
    
        outliner_layout = QtWidgets.QVBoxLayout()
        outliner_layout.setContentsMargins(6, 6, 6, 6)
        outliner_layout.setSpacing(6)
    
        # QTreeWidget replaces QListWidget
        tree = QtWidgets.QTreeWidget()
        tree.setHeaderHidden(True)
        tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        tree.setExpandsOnDoubleClick(True)
        tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
    
        # Hook signals
        tree.itemSelectionChanged.connect(self.on_tree_selection_changed)
        tree.customContextMenuRequested.connect(self.show_outliner_context_menu)
    
        # Keep compatibility: reuse attribute name
        self.scene_outliner = tree
    
        outliner_layout.addWidget(tree)
        outliner_group.setLayout(outliner_layout)
    
        left_layout.addWidget(outliner_group, 1)
        left_container.setLayout(left_layout)
        self.left_dock.setWidget(left_container)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.left_dock)
    
        # Ensure a default top-level "Collection" exists
        self.ensure_collection("Collection")
    
        # RIGHT DOCK (unchanged)
        self.right_dock = QtWidgets.QDockWidget("Properties & Details", self)
        self.right_dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea)
        right_container = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(6)
    
        properties_group = QtWidgets.QGroupBox("Properties")
        properties_layout = QtWidgets.QVBoxLayout()
    
        self.tabs = QtWidgets.QTabWidget()
        self.transform_tab, self.appearance_tab, self.lighting_tab, self.details_tab = QtWidgets.QWidget(), QtWidgets.QWidget(), QtWidgets.QWidget(), QtWidgets.QWidget()
        self.tabs.addTab(self.transform_tab, "Transform")
        self.tabs.addTab(self.appearance_tab, "Appearance")
        self.tabs.addTab(self.lighting_tab, "Lighting")
        self.tabs.addTab(self.details_tab, "Details")
    
        # Build the tab UIs
        self.setup_transform_tab()
        self.setup_appearance_tab()
        self.setup_lighting_tab()
        self.setup_details_tab()
    
        properties_layout.addWidget(self.tabs)
        properties_group.setLayout(properties_layout)
    
        right_layout.addWidget(properties_group)
        right_container.setLayout(right_layout)
        self.right_dock.setWidget(right_container)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.right_dock)
    
        # Initial dock sizes
        try:
            self.resizeDocks([self.left_dock, self.right_dock], [300, 360], QtCore.Qt.Horizontal)
        except Exception:
            pass
    
        # Install a simple viewport picker to sync viewport -> tree selection
        self.install_viewport_picker()
    def show_outliner_context_menu(self, position):
        """Show right-click context menu for scene outliner items."""
        item = self.scene_outliner.itemAt(position)
        if item:
            context_menu = QtWidgets.QMenu(self)
            # NEW: copy/paste in context menu
            copy_act = QtWidgets.QAction("Copy", self)
            copy_act.setShortcut("Ctrl+C")
            copy_act.triggered.connect(self.on_copy_selected)
            paste_act = QtWidgets.QAction("Paste (Duplicate)", self)
            paste_act.setShortcut("Ctrl+V")
            paste_act.triggered.connect(self.on_paste_selected)
            context_menu.addAction(copy_act)
            context_menu.addAction(paste_act)
            context_menu.addSeparator()
            delete_action = QtWidgets.QAction("Delete", self)
            delete_action.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_TrashIcon))
            delete_action.triggered.connect(lambda: self.delete_selected_object(item))
            context_menu.addAction(delete_action)
            context_menu.exec_(self.scene_outliner.mapToGlobal(position))

    def delete_selected_object(self, item):
        """Delete the selected scene entry (mesh or light). Mesh deletions are undoable."""
        if not item:
            return
        name = item.text(0) if isinstance(item, QtWidgets.QTreeWidgetItem) else item.text()

        reply = QtWidgets.QMessageBox.question(
            self, 'Delete',
            f'Are you sure you want to delete "{name}"?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        # Light deletion (not yet undoable)
        if name in self.light_registry:
            entry = self.light_registry[name]
            if entry.get('gizmo'):
                try:
                    self.vtk_app.renderer.RemoveActor(entry['gizmo'])
                    if entry['gizmo'] in self.vtk_app.actors:
                        self.vtk_app.actors.remove(entry['gizmo'])
                except Exception:
                    pass
            self.vtk_app.remove_light_from_scene(entry['light'])
            self.light_registry.pop(name, None)

            # remove tree item
            if isinstance(item, QtWidgets.QTreeWidgetItem):
                parent = item.parent()
                if parent:
                    parent.removeChild(item)
                else:
                    idx = self.scene_outliner.indexOfTopLevelItem(item)
                    self.scene_outliner.takeTopLevelItem(idx)

            # clear gizmo if needed
            if self.transform_widget:
                try:
                    self.transform_widget.Off()
                except Exception:
                    pass
                self.transform_widget = None

            self.update_properties_panel(None)
            self.update_scene_totals()
            self.vtk_app.render_all()
            self.statusBar().showMessage(f'Deleted "{name}"')
            return

        # Mesh actor deletion (undoable)
        actor = self.object_registry.get(name)
        if actor:
            self.undo_stack.push(DeleteActorCommand(self, name, actor))
            return

        self.statusBar().showMessage("Delete: item not found")

    def ensure_collection(self, name: str) -> QtWidgets.QTreeWidgetItem:
        """Get or create a top-level Collection node."""
        if not isinstance(self.scene_outliner, QtWidgets.QTreeWidget):
            return None
        # Find existing
        for i in range(self.scene_outliner.topLevelItemCount()):
            it = self.scene_outliner.topLevelItem(i)
            if it.text(0) == name and it.data(0, QtCore.Qt.UserRole) == "collection":
                return it
        # Create new
        item = QtWidgets.QTreeWidgetItem([name])
        item.setData(0, QtCore.Qt.UserRole, "collection")
        item.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_DirIcon))
        item.setFlags(item.flags() | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        self.scene_outliner.addTopLevelItem(item)
        item.setExpanded(True)
        return item
    
    def _make_object_item(self, unique_name: str, kind: str, parent: QtWidgets.QTreeWidgetItem = None) -> QtWidgets.QTreeWidgetItem:
        """Create a tree item for a mesh/light/camera under parent (collection)."""
        if not isinstance(self.scene_outliner, QtWidgets.QTreeWidget):
            return None
        parent = parent or self.ensure_collection("Collection")
        item = QtWidgets.QTreeWidgetItem([unique_name])
        item.setData(0, QtCore.Qt.UserRole, kind)  # 'mesh' | 'light' | 'camera'
        # Icons
        if kind == "mesh":
            item.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon))
        elif kind == "light":
            item.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_DialogYesButton))
        else:
            item.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_DesktopIcon))
        parent.addChild(item)
        return item
    
    def on_tree_selection_changed(self):
        """Tree -> viewport selection sync (use first selected object)."""
        items = self.scene_outliner.selectedItems()
        if not items:
            self.update_properties_panel(None)
            self.update_scene_totals()
            return
        # Pick first non-collection item if multiple
        chosen = None
        for it in items:
            if it.data(0, QtCore.Qt.UserRole) in ("mesh", "light"):
                chosen = it
                break
        if not chosen:
            # Only collection(s) selected
            self.update_properties_panel(None)
            self.update_scene_totals()
            return
        # Reuse existing handler
        self.on_outliner_selection_changed(chosen)
    
    def show_outliner_context_menu(self, position):
        """Right-click menu for tree items (delete, copy/paste)."""
        item = self.scene_outliner.itemAt(position)
        if not item:
            return
        kind = item.data(0, QtCore.Qt.UserRole)
    
        menu = QtWidgets.QMenu(self)
    
        # Copy/Paste for objects only
        if kind in ("mesh", "light"):
            copy_act = QtWidgets.QAction("Copy", self); copy_act.setShortcut("Ctrl+C")
            copy_act.triggered.connect(self.on_copy_selected)
            paste_act = QtWidgets.QAction("Paste (Duplicate)", self); paste_act.setShortcut("Ctrl+V")
            paste_act.triggered.connect(self.on_paste_selected)
            menu.addAction(copy_act); menu.addAction(paste_act); menu.addSeparator()
    
        delete_action = QtWidgets.QAction("Delete", self)
        delete_action.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_TrashIcon))
    
        def do_delete():
            k = item.data(0, QtCore.Qt.UserRole)
            if k == "collection":
                # Confirm cascade delete
                reply = QtWidgets.QMessageBox.question(
                    self, "Delete Collection",
                    f'Delete collection "{item.text(0)}" and all children?',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No
                )
                if reply != QtWidgets.QMessageBox.Yes:
                    return
                self._delete_collection_recursive(item)
            else:
                # Reuse existing per-item delete with undo for meshes
                self.delete_selected_object(item)
    
        delete_action.triggered.connect(do_delete)
        menu.addAction(delete_action)
        menu.exec_(self.scene_outliner.viewport().mapToGlobal(position))
    
    def _delete_collection_recursive(self, col_item: QtWidgets.QTreeWidgetItem):
        """Delete all child mesh/lights under a collection, then the collection."""
        # Collect children first (stable list)
        to_delete = []
        def walk(it):
            for i in range(it.childCount()):
                ch = it.child(i)
                kd = ch.data(0, QtCore.Qt.UserRole)
                if kd == "collection":
                    walk(ch)
                else:
                    to_delete.append(ch)
        walk(col_item)
        # Delete children using existing path
        for obj_item in to_delete:
            self.delete_selected_object(obj_item)
        # Remove empty collection node
        parent = col_item.parent()
        if parent:
            parent.removeChild(col_item)
        else:
            idx = self.scene_outliner.indexOfTopLevelItem(col_item)
            self.scene_outliner.takeTopLevelItem(idx)
        self.vtk_app.render_all()
    
    def install_viewport_picker(self):
        """Simple click-pick to select object in tree."""
        if not self.vtk_app or not self.vtk_app.interactor:
            return
        self._tree_picker = vtk.vtkCellPicker()
        self._tree_picker.SetTolerance(0.0005)
    
        def on_click(obj, evt):
            pos = self.vtk_app.interactor.GetEventPosition()
            self._tree_picker.Pick(pos[0], pos[1], 0, self.vtk_app.renderer)
            act = self._tree_picker.GetActor()
            if not act:
                return
            # Find matching name in registry
            name = None
            for n, a in self.object_registry.items():
                if a is act:
                    name = n; break
            if not name:
                # Maybe a light gizmo
                for n, entry in self.light_registry.items():
                    if entry.get("gizmo") is act:
                        name = n; break
            if not name:
                return
            # Locate item in tree and select
            it = self._find_tree_item_by_name(name)
            if it:
                self.scene_outliner.clearSelection()
                it.setSelected(True)
                self.scene_outliner.scrollToItem(it)
                self.on_outliner_selection_changed(it)
    
        # Install only once
        if not hasattr(self, "_tree_pick_tag") or self._tree_pick_tag is None:
            self._tree_pick_tag = self.vtk_app.interactor.AddObserver("LeftButtonReleaseEvent", on_click)
    
    def _find_tree_item_by_name(self, name: str) -> QtWidgets.QTreeWidgetItem:
        """Find a tree item by its text (depth-first)."""
        def dfs(it):
            if it.text(0) == name:
                return it
            for i in range(it.childCount()):
                r = dfs(it.child(i))
                if r: return r
            return None
        for i in range(self.scene_outliner.topLevelItemCount()):
            r = dfs(self.scene_outliner.topLevelItem(i))
            if r: return r
        return None

    # ===== Camera Mode (orbit/pan/zoom) =====
    def on_camera_mode_toggled(self, checked: bool):
        if checked:
            self.enter_camera_mode()
        else:
            self.exit_camera_mode()

    def enter_camera_mode(self):
        if self.camera_mode:
            return
        # Stop tools/gizmos to avoid conflicts
        try:
            if self.current_tool:
                self.current_tool.stop(cancel=True)
                self.current_tool = None
        except Exception:
            pass
        try:
            if self.transform_widget:
                self.transform_widget.Off()
                self.transform_widget = None
        except Exception:
            pass
        # Save and switch interactor style
        self._prev_interactor_style = self.vtk_app.interactor.GetInteractorStyle()
        try:
            cam_style = vtk.vtkInteractorStyleTrackballCamera()
            self.vtk_app.interactor.SetInteractorStyle(cam_style)
        except Exception:
            pass
        # Badge
        self._show_camera_mode_label()
        # Observers: arrows pan; Shift+Wheel = horizontal pan; Esc exits
        self._camera_add_obs("KeyPressEvent", self._camera_keypress_cb)
        self._camera_add_obs("MouseWheelForwardEvent", self._camera_wheel_forward_cb)
        self._camera_add_obs("MouseWheelBackwardEvent", self._camera_wheel_backward_cb)
        self.camera_mode = True
        self.statusBar().showMessage("Camera Mode: orbit/pan/zoom enabled. Esc to exit.")

    def exit_camera_mode(self):
        if not self.camera_mode:
            return
        # Clear observers
        self._camera_clear_obs()
        # Restore style
        try:
            if self._prev_interactor_style:
                self.vtk_app.interactor.SetInteractorStyle(self._prev_interactor_style)
        except Exception:
            pass
        self._prev_interactor_style = None
        # Hide badge
        self._hide_camera_mode_label()
        self.camera_mode = False
        if self.camera_mode_action.isChecked():
            self.camera_mode_action.setChecked(False)
        self.statusBar().showMessage("Camera Mode: off")
        self.vtk_app.render_all()

    def _show_camera_mode_label(self):
        if self.camera_mode_label:
            self.camera_mode_label.show()
            if hasattr(self, 'camera_reset_button'):
                self.camera_reset_button.show()
            return
        self.camera_mode_label = QtWidgets.QLabel("CAMERA MODE", self.vtkWidget)
        self.camera_mode_label.setStyleSheet(
            "QLabel{background:rgba(20,20,20,160); color:#ffd166; "
            "border:1px solid #555; padding:4px 8px; font: 10pt 'Consolas';}"
        )
        self.camera_mode_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        
        # NEW: Reset Camera button
        self.camera_reset_button = QtWidgets.QPushButton("Reset Camera", self.vtkWidget)
        self.camera_reset_button.setStyleSheet(
            "QPushButton{background:rgba(30,30,30,180); color:#ffd166; "
            "border:1px solid #555; padding:4px 8px; font: 9pt 'Consolas';}"
            "QPushButton:hover{background:rgba(50,50,50,200);}"
            "QPushButton:pressed{background:rgba(20,20,20,200);}"
        )
        self.camera_reset_button.clicked.connect(self.reset_camera_to_default)
        
        self._position_camera_label()
        self.camera_mode_label.show()
        self.camera_reset_button.show()

    def reset_camera_to_default(self):
        """Reset camera to the default Blender-like position."""
        camera = self.vtk_app.renderer.GetActiveCamera()
        
        # Default position from setup_default_camera
        camera.SetPosition(20, -20, 15)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 0, 1)
        camera.SetParallelProjection(False)
        camera.SetViewAngle(40)
        
        self.vtk_app.renderer.ResetCameraClippingRange()
        self.vtk_app.render_all()
        self.statusBar().showMessage("Camera reset to default position")

    def _hide_camera_mode_label(self):
        if self.camera_mode_label:
            self.camera_mode_label.hide()
        if hasattr(self, 'camera_reset_button'):
            self.camera_reset_button.hide()

    def _position_camera_label(self):
        if not self.camera_mode_label:
            return
        # top-right corner under the axes region
        margin = 10
        w = self.vtkWidget.width()
        self.camera_mode_label.adjustSize()
        lw = self.camera_mode_label.width()
        lh = self.camera_mode_label.height()
        x_pos = max(0, w - lw - margin)
        y_pos = margin
        self.camera_mode_label.move(x_pos, y_pos)
        
        # Position button below the label
        if hasattr(self, 'camera_reset_button'):
            self.camera_reset_button.adjustSize()
            bw = self.camera_reset_button.width()
            bh = self.camera_reset_button.height()
            button_x = max(0, w - bw - margin)
            button_y = y_pos + lh + 4  # 4px spacing below label
            self.camera_reset_button.move(button_x, button_y)

    def _camera_add_obs(self, evt, cb):
        try:
            tag = self.vtk_app.interactor.AddObserver(evt, cb, 1.0)
            self._camera_obs.append((evt, tag))
        except Exception:
            pass

    def _camera_clear_obs(self):
        for (_, tag) in self._camera_obs:
            try:
                self.vtk_app.interactor.RemoveObserver(tag)
            except Exception:
                pass
        self._camera_obs.clear()

    def _camera_keypress_cb(self, obj, evt):
        key = self.vtk_app.interactor.GetKeySym() or ""
        if key == "Escape":
            # Abort further handling and exit
            try:
                obj.SetAbortFlag(1)
            except Exception:
                pass
            self.exit_camera_mode()
            return
        if key in ("Left", "Right", "Up", "Down"):
            # Pan step based on distance to focal
            cam = self.vtk_app.renderer.GetActiveCamera()
            px, py, pz = cam.GetPosition()
            fx, fy, fz = cam.GetFocalPoint()
            import math
            dist = math.sqrt((px - fx)**2 + (py - fy)**2 + (pz - fz)**2) or 1.0
            step = dist * (0.05 if not self.vtk_app.interactor.GetShiftKey() else 0.15)
            # Camera vectors
            right = self._camera_right_vector(cam)
            up = cam.GetViewUp()
            dx = dy = dz = 0.0
            if key == "Left":
                dx, dy, dz = (-right[0]*step, -right[1]*step, -right[2]*step)
            elif key == "Right":
                dx, dy, dz = (right[0]*step, right[1]*step, right[2]*step)
            elif key == "Up":
                dx, dy, dz = (up[0]*step, up[1]*step, up[2]*step)
            elif key == "Down":
                dx, dy, dz = (-up[0]*step, -up[1]*step, -up[2]*step)
            cam.SetPosition(px + dx, py + dy, pz + dz)
            cam.SetFocalPoint(fx + dx, fy + dy, fz + dz)
            self.vtk_app.renderer.ResetCameraClippingRange()
            self.vtk_app.render_all()
            try:
                obj.SetAbortFlag(1)
            except Exception:
                pass

    def _camera_wheel_forward_cb(self, obj, evt):
        self._camera_handle_wheel(obj, +1)

    def _camera_wheel_backward_cb(self, obj, evt):
        self._camera_handle_wheel(obj, -1)

    def _camera_handle_wheel(self, obj, sign):
        # Shift + Wheel => horizontal pan; else let style handle default zoom
        if not self.vtk_app.interactor.GetShiftKey():
            return
        cam = self.vtk_app.renderer.GetActiveCamera()
        px, py, pz = cam.GetPosition()
        fx, fy, fz = cam.GetFocalPoint()
        import math
        dist = math.sqrt((px - fx)**2 + (py - fy)**2 + (pz - fz)**2) or 1.0
        step = dist * 0.08 * sign
        right = self._camera_right_vector(cam)
        dx, dy, dz = (right[0]*step, right[1]*step, right[2]*step)
        cam.SetPosition(px + dx, py + dy, pz + dz)
        cam.SetFocalPoint(fx + dx, fy + dy, fz + dz)
        self.vtk_app.renderer.ResetCameraClippingRange()
        self.vtk_app.render_all()
        # Prevent default zoom on this wheel event
        try:
            obj.SetAbortFlag(1)
        except Exception:
            pass

    def _camera_right_vector(self, camera):
        # right = normalize(cross(viewDir, viewUp))
        vx, vy, vz = camera.GetDirectionOfProjection()
        ux, uy, uz = camera.GetViewUp()
        rx = vy*uz - vz*uy
        ry = vz*ux - vx*uz
        rz = vx*uy - vy*ux
        import math
        mag = math.sqrt(rx*rx + ry*ry + rz*rz) or 1.0
        return (rx/mag, ry/mag, rz/mag)
    # ===== end Camera Mode =====

    def add_new_light(self, light_type: str):
        """Create a light + gizmo actor and add to scene and collection tree."""
        light = self.vtk_app.create_light(light_type)
        if not light:
            return
        self.vtk_app.add_light_to_scene(light)
    
        # Gizmo sphere
        src = vtk.vtkSphereSource(); src.SetRadius(0.6)
        mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(src.GetOutputPort())
        gizmo_actor = vtk.vtkActor(); gizmo_actor.SetMapper(mapper)
        gizmo_actor.GetProperty().SetColor(1.0, 0.9, 0.2); gizmo_actor.GetProperty().SetOpacity(0.9)
        gizmo_actor.PickableOn()
        gizmo_actor.SetPosition(*light.GetPosition())
    
        base = f"{light_type.lower()}_light"
        count = len([n for n in list(self.light_registry.keys()) if n.startswith(base)]) + 1
        name = f"{base}_{count}"
        self.light_registry[name] = {'light': light, 'gizmo': gizmo_actor, 'type': light_type}
        self.vtk_app.add_actor_to_scene(gizmo_actor)
    
        # Add in tree under selected collection or default
        parent = None
        if isinstance(self.scene_outliner, QtWidgets.QTreeWidget):
            sel = self.scene_outliner.selectedItems()
            if sel and sel[0].data(0, QtCore.Qt.UserRole) == "collection":
                parent = sel[0]
            else:
                parent = self.ensure_collection("Collection")
            item = self._make_object_item(name, "light", parent)
            self.scene_outliner.setCurrentItem(item)
            self.on_outliner_selection_changed(item)
        else:
            item = QtWidgets.QListWidgetItem(name, self.scene_outliner)
            item.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogYesButton))
            self.scene_outliner.setCurrentItem(item)
            self.on_outliner_selection_changed(item)

    def setup_transform_tab(self):
        layout, self.spinboxes = QtWidgets.QFormLayout(), {}
        for t_type in ["Position", "Rotation", "Scale"]:
            for axis in "XYZ":
                key = f"{t_type}{axis}"
                self.spinboxes[key] = QtWidgets.QDoubleSpinBox()
                self.spinboxes[key].setRange(-1000, 1000)
                self.spinboxes[key].setDecimals(3)
                self.spinboxes[key].setSingleStep(0.5 if t_type != "Scale" else 0.1)
                self.spinboxes[key].valueChanged.connect(self.on_transform_changed)
                layout.addRow(f"{t_type} {axis}", self.spinboxes[key])

        # Reset row (clean 2x2 grid)
        reset_row = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(reset_row)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)

        btn_reset_loc = QtWidgets.QPushButton("Reset Location")
        btn_reset_rot = QtWidgets.QPushButton("Reset Rotation")
        btn_reset_scl = QtWidgets.QPushButton("Reset Scale")
        btn_reset_all = QtWidgets.QPushButton("Reset All")

        btn_reset_loc.clicked.connect(lambda: self.on_reset_transform("loc"))
        btn_reset_rot.clicked.connect(lambda: self.on_reset_transform("rot"))
        btn_reset_scl.clicked.connect(lambda: self.on_reset_transform("scale"))
        btn_reset_all.clicked.connect(lambda: self.on_reset_transform("all"))

        grid.addWidget(btn_reset_loc, 0, 0)
        grid.addWidget(btn_reset_rot, 0, 1)
        grid.addWidget(btn_reset_scl, 1, 0)
        grid.addWidget(btn_reset_all, 1, 1)

        layout.addRow(reset_row)

        # Separator before edit mode button
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addRow(sep)

        # Edit mode section
        edit_row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(edit_row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)

        self.vertex_edit_button = QtWidgets.QPushButton("Enter Edit Mode")
        self.vertex_edit_button.setCheckable(True)
        self.vertex_edit_button.toggled.connect(self.on_vertex_edit_toggled)
        h.addWidget(self.vertex_edit_button, 1)

        # Sub-mode buttons (appear only when Edit Mode is ON)
        self.edit_vertex_mode_btn = QtWidgets.QPushButton("Edit Vertex")
        self.edit_vertex_mode_btn.setCheckable(True)
        self.edit_face_mode_btn = QtWidgets.QPushButton("Edit Face")
        self.edit_face_mode_btn.setCheckable(True)
        self.edit_vertex_mode_btn.clicked.connect(lambda: self._switch_edit_submode("vertex"))
        self.edit_face_mode_btn.clicked.connect(lambda: self._switch_edit_submode("face"))
        # hidden by default
        self.edit_vertex_mode_btn.hide()
        self.edit_face_mode_btn.hide()
        h.addWidget(self.edit_vertex_mode_btn, 0)
        h.addWidget(self.edit_face_mode_btn, 0)

        layout.addRow(edit_row)

        self.transform_tab.setLayout(layout)

    def on_reset_transform(self, which: str):
        """Reset transform components for the current selection."""
        # Lights: only position is meaningful; preserve direction vector
        light, lname = self.get_selected_light()
        if light:
            if which in ("loc", "all"):
                # Keep current direction
                px0, py0, pz0 = light.GetPosition()
                fx, fy, fz = light.GetFocalPoint()
                dir_vec = (fx - px0, fy - py0, fz - pz0)

                new_pos = (0.0, 0.0, 0.0)
                light.SetPosition(*new_pos)
                light.SetFocalPoint(new_pos[0] + dir_vec[0],
                                    new_pos[1] + dir_vec[1],
                                    new_pos[2] + dir_vec[2])
                # Move gizmo and sync UI
                giz = self.light_registry[lname]['gizmo']
                giz.SetPosition(*new_pos)
                for i, ax in enumerate("XYZ"):
                    self.spinboxes[f"Position{ax}"].setValue(new_pos[i])

                self._populate_light_controls(light, self.light_registry[lname]['type'])
                self.vtk_app.render_all()
            return

        # Mesh actor branch
        actor = self.get_selected_actor()
        if not actor:
            return

        before = self._get_actor_user_matrix16(actor)

        # Update spinboxes without spamming valueChanged
        self.block_signals = True
        if which in ("loc", "all"):
            for ax in "XYZ":
                self.spinboxes[f"Position{ax}"].setValue(0.0)
        if which in ("rot", "all"):
            for ax in "XYZ":
                self.spinboxes[f"Rotation{ax}"].setValue(0.0)
        if which in ("scale", "all"):
            for ax in "XYZ":
                self.spinboxes[f"Scale{ax}"].setValue(1.0)
        self.block_signals = False

        # Apply once using existing handler
        self.on_transform_changed()

        after = self._get_actor_user_matrix16(actor)
        if after != before:
            self.undo_stack.push(TransformActorCommand(self, actor, before, after))

    def setup_appearance_tab(self):
        layout, self.sliders, self.combos = QtWidgets.QFormLayout(), {}, {}
        self.slider_value_labels = {}
        self.checks = {}

        def make_slider_row(name, min_v, max_v, start_v, show_percent=True):
            sld = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            sld.setRange(min_v, max_v)
            sld.setValue(start_v)
            row = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            h.addWidget(sld)
            lbl = QtWidgets.QLabel(f"{start_v}%" if show_percent else str(start_v))
            lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            lbl.setFixedWidth(60)
            self.slider_value_labels[name] = lbl
            h.addWidget(lbl)

            def on_changed(val):
                lbl.setText(f"{val}%" if show_percent else str(val))
                self.on_appearance_changed()

            sld.valueChanged.connect(on_changed)
            # Undo grouping hooks
            sld.sliderPressed.connect(self.on_prop_slider_pressed)
            sld.sliderReleased.connect(self.on_prop_slider_released)

            self.sliders[name] = sld
            layout.addRow(name, row)

        # Simple color dialog button
        color_button = QtWidgets.QPushButton("Change Color")
        color_button.clicked.connect(self.change_current_object_color)
        layout.addRow(color_button)

        # Core material sliders
        make_slider_row("Opacity",        0, 100, 100, True)
        make_slider_row("Ambient",        0, 100,   5, True)
        make_slider_row("Diffuse",        0, 100, 100, True)
        make_slider_row("Specular",       0, 100,  30, True)
        make_slider_row("SpecularPower",  1, 256,  64, False)

        # Shading
        self.combos["Interpolation"] = QtWidgets.QComboBox()
        self.combos["Interpolation"].addItems(["Flat", "Gouraud", "Phong"])
        self.combos["Interpolation"].setCurrentIndex(1)
        self.combos["Interpolation"].currentIndexChanged.connect(self.on_appearance_changed)
        layout.addRow("Shading", self.combos["Interpolation"])

        # ---- NEW: Color Texture section -------------------------------------------------
        tex_row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(tex_row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)

        # Thumbnail square
        self.tex_thumb_label = QtWidgets.QLabel()
        self.tex_thumb_label.setFixedSize(48, 48)
        self.tex_thumb_label.setFrameShape(QtWidgets.QFrame.Box)
        self.tex_thumb_label.setLineWidth(1)
        self.tex_thumb_label.setAlignment(QtCore.Qt.AlignCenter)
        self.tex_thumb_label.setStyleSheet("background: #2b2b2b;")
        self.tex_thumb_label.setToolTip("No texture")
        self.tex_thumb_label.installEventFilter(self)  # clickable

        self.tex_load_button = QtWidgets.QPushButton("Load…")
        self.tex_load_button.clicked.connect(self.on_load_texture_clicked)

        self.tex_clear_button = QtWidgets.QPushButton("Clear")
        self.tex_clear_button.clicked.connect(self.on_clear_texture_clicked)
        self.tex_clear_button.setEnabled(False)

        h.addWidget(self.tex_thumb_label, 0)
        h.addStretch(1)
        h.addWidget(self.tex_load_button, 0)
        h.addWidget(self.tex_clear_button, 0)

        layout.addRow("Image Texture", tex_row)
        # -------------------------------------------------------------------------------

        # Separator
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addRow(sep)

        # Render mode
        self.combos["Representation"] = QtWidgets.QComboBox()
        self.combos["Representation"].addItems(["Surface", "Wireframe", "Points"])
        self.combos["Representation"].setCurrentIndex(0)
        self.combos["Representation"].currentIndexChanged.connect(self.on_appearance_changed)
        layout.addRow("Render Mode", self.combos["Representation"])

        # Toggles
        self.checks["ShowEdges"] = QtWidgets.QCheckBox("Show Edges")
        self.checks["ShowEdges"].stateChanged.connect(self.on_appearance_changed)
        layout.addRow("", self.checks["ShowEdges"])

        self.checks["BackfaceCulling"] = QtWidgets.QCheckBox("Backface Culling")
        self.checks["BackfaceCulling"].stateChanged.connect(self.on_appearance_changed)
        layout.addRow("", self.checks["BackfaceCulling"])

        self.checks["FrontfaceCulling"] = QtWidgets.QCheckBox("Frontface Culling")
        self.checks["FrontfaceCulling"].stateChanged.connect(self.on_appearance_changed)
        layout.addRow("", self.checks["FrontfaceCulling"])

        self.appearance_tab.setLayout(layout)

    def _get_actor_property_snapshot(self, actor: vtk.vtkActor):
        prop = actor.GetProperty()
        return {
            "color": prop.GetColor(),
            "opacity": prop.GetOpacity(),
            "ambient": prop.GetAmbient(),
            "diffuse": prop.GetDiffuse(),
            "specular": prop.GetSpecular(),
            "specularPower": prop.GetSpecularPower(),
            "interp": prop.GetInterpolation(),
            "repr": prop.GetRepresentation(),
            "edgeVis": int(prop.GetEdgeVisibility()),
            "backCull": int(prop.GetBackfaceCulling()),
            "frontCull": int(prop.GetFrontfaceCulling())
        }

    def _apply_actor_property_snapshot(self, actor: vtk.vtkActor, snap: dict):
        prop = actor.GetProperty()
        prop.SetColor(snap["color"])
        prop.SetOpacity(snap["opacity"])
        prop.SetAmbient(snap["ambient"])
        prop.SetDiffuse(snap["diffuse"])
        prop.SetSpecular(snap["specular"])
        prop.SetSpecularPower(snap["specularPower"])
        prop.SetInterpolation(snap["interp"])
        # Representation: 2 surface / 1 wire / 0 points
        rep = snap["repr"]
        if rep == 2: prop.SetRepresentationToSurface()
        elif rep == 1: prop.SetRepresentationToWireframe()
        else: prop.SetRepresentationToPoints()
        prop.SetEdgeVisibility(bool(snap["edgeVis"]))
        if snap["backCull"]: prop.BackfaceCullingOn()
        else: prop.BackfaceCullingOff()
        if snap["frontCull"]: prop.FrontfaceCullingOn()
        else: prop.FrontfaceCullingOff()

    def on_prop_slider_pressed(self):
        if self.block_signals: return
        actor = self.get_selected_actor()
        light, _ = self.get_selected_light()
        if not actor or light: return
        # Begin a grouped slider change
        self._pending_prop_snapshot = self._get_actor_property_snapshot(actor)

    def on_prop_slider_released(self):
        if self.block_signals: return
        actor = self.get_selected_actor()
        light, _ = self.get_selected_light()
        if not actor or light or not self._pending_prop_snapshot: 
            self._pending_prop_snapshot = None
            return
        after = self._get_actor_property_snapshot(actor)
        if after != self._pending_prop_snapshot:
            self.undo_stack.push(PropertyChangeCommand(self, actor, self._pending_prop_snapshot, after))
        self._pending_prop_snapshot = None

     # ----- helpers for uniform scale (Shift) -----
    def _install_scale_key_observers(self):
        if self._scale_obs or not self.vtk_app.interactor:
            return
        def on_keypress(obj, evt):
            key = self.vtk_app.interactor.GetKeySym() or ""
            if key in ("Shift_L", "Shift_R"):
                self._scale_uniform_lock = True
                self._ensure_uniform_scale_hints()
                self._update_uniform_scale_hints(self.current_selected_actor)
        def on_keyrelease(obj, evt):
            key = self.vtk_app.interactor.GetKeySym() or ""
            if key in ("Shift_L", "Shift_R"):
                self._scale_uniform_lock = False
                self._update_uniform_scale_hints(self.current_selected_actor)
        def on_render(*args):
            self._update_uniform_scale_hints(self.current_selected_actor)
        iren = self.vtk_app.interactor
        self._scale_obs.append(("KeyPressEvent",  iren.AddObserver("KeyPressEvent",  on_keypress, 1.0)))
        self._scale_obs.append(("KeyReleaseEvent",iren.AddObserver("KeyReleaseEvent", on_keyrelease,1.0)))
        self._scale_obs.append(("RenderEvent",    iren.AddObserver("RenderEvent",    on_render,   0.0)))

    def _remove_scale_key_observers(self):
        if not self._scale_obs or not self.vtk_app.interactor:
            self._scale_obs = []
            return
        for (_, tag) in self._scale_obs:
            try:
                self.vtk_app.interactor.RemoveObserver(tag)
            except Exception:
                pass
        self._scale_obs.clear()
        self._scale_uniform_lock = False

    def _ensure_uniform_scale_hints(self):
        if self._uniform_hint_actors:
            return
        # Two billboard text actors (left/right)
        for _ in range(2):
            txt = vtk.vtkBillboardTextActor3D()
            txt.SetInput("⇔")
            tp = txt.GetTextProperty()
            tp.SetFontSize(32)
            tp.BoldOn()
            tp.SetColor(1.0, 0.9, 0.2)
            txt.VisibilityOff()
            self.vtk_app.renderer.AddActor(txt)
            self._uniform_hint_actors.append(txt)

    def _clear_uniform_scale_hints(self):
        if not self._uniform_hint_actors:
            return
        for a in self._uniform_hint_actors:
            try:
                self.vtk_app.renderer.RemoveActor(a)
            except Exception:
                pass
        self._uniform_hint_actors = []

    def _update_uniform_scale_hints(self, actor):
        # Visible only in Scale mode and while Shift is held
        show = (self.current_transform_mode == 'scale' and
                self._scale_uniform_lock and
                self.transform_widget is not None and actor is not None)
        if not self._uniform_hint_actors:
            return
        for a in self._uniform_hint_actors:
            a.SetVisibility(show)
        if not show:
            self.vtk_app.render_all()
            return
        # Place hints on left/right sides of the actor bounds
        try:
            xmin, xmax, ymin, ymax, zmin, zmax = actor.GetBounds()
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            cz = 0.5 * (zmin + zmax)
            dx = max(1e-6, xmax - xmin)
            dy = max(1e-6, ymax - ymin)
            dz = max(1e-6, zmax - zmin)
            s = 0.08 * max(dx, dy, dz)  # text scale vs object size
            self._uniform_hint_actors[0].SetPosition(xmin, cy, cz)
            self._uniform_hint_actors[1].SetPosition(xmax, cy, cz)
            self._uniform_hint_actors[0].SetScale(s, s, s)
            self._uniform_hint_actors[1].SetScale(s, s, s)
        except Exception:
            pass

    def _make_uniform_scale_transform(self, tf: vtk.vtkTransform) -> vtk.vtkTransform:
        """Return a copy of tf with uniform scale (averaged) and same translation."""
        m = vtk.vtkMatrix4x4()
        tf.GetMatrix(m)
        # Extract per-axis scales (rotation is disabled in scale mode)
        sx = abs(m.GetElement(0, 0))
        sy = abs(m.GetElement(1, 1))
        sz = abs(m.GetElement(2, 2))
        if sx == 0: sx = 1.0
        if sy == 0: sy = 1.0
        if sz == 0: sz = 1.0
        s = (sx + sy + sz) / 3.0
        new_m = vtk.vtkMatrix4x4()
        new_m.Identity()
        new_m.SetElement(0, 0, s)
        new_m.SetElement(1, 1, s)
        new_m.SetElement(2, 2, s)
        # keep translation
        new_m.SetElement(0, 3, m.GetElement(0, 3))
        new_m.SetElement(1, 3, m.GetElement(1, 3))
        new_m.SetElement(2, 3, m.GetElement(2, 3))
        out = vtk.vtkTransform()
        out.SetMatrix(new_m)
        return out


    def setup_lighting_tab(self):
        """Creates the UI for the Lighting tab with RGB color control."""
        lg = QtWidgets.QFormLayout()

        self.lighting_controls = {}
        self.lighting_slider_value_labels = {}

        # Light Type
        self.lighting_controls["Type"] = QtWidgets.QComboBox()
        self.lighting_controls["Type"].addItems(["Point", "Directional", "Spot"])
        self.lighting_controls["Type"].currentIndexChanged.connect(self.on_light_controls_changed)
        lg.addRow("Type", self.lighting_controls["Type"])

        # Position XYZ
        for ax in "XYZ":
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(-1000, 1000)
            sb.setDecimals(3)
            sb.valueChanged.connect(self.on_light_controls_changed)
            self.lighting_controls[f"Pos{ax}"] = sb
            lg.addRow(f"Position {ax}", sb)

        # Direction XYZ
        for ax in "XYZ":
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(-1e3, 1e3)
            sb.setDecimals(3)
            sb.valueChanged.connect(self.on_light_controls_changed)
            self.lighting_controls[f"Dir{ax}"] = sb
            lg.addRow(f"Direction {ax}", sb)

        # Intensity
        self.lighting_controls["Intensity"] = QtWidgets.QDoubleSpinBox()
        self.lighting_controls["Intensity"].setRange(0.0, 50.0)
        self.lighting_controls["Intensity"].setSingleStep(0.1)
        self.lighting_controls["Intensity"].valueChanged.connect(self.on_light_controls_changed)
        lg.addRow("Intensity", self.lighting_controls["Intensity"])

        # RGB Color Sliders (replacing Ambient/Diffuse/Specular)
        def make_light_slider(name):
            sld = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            sld.setRange(0, 100)
            sld.setValue(100)  # Default to white (100%)
            sld.valueChanged.connect(self.on_light_controls_changed)
            row = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(sld)
            lbl = QtWidgets.QLabel("100%")
            lbl.setFixedWidth(50)
            lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            h.addWidget(lbl)
            self.lighting_controls[name] = sld
            self.lighting_slider_value_labels[name] = lbl
            lg.addRow(name, row)

        for name in ["Red", "Green", "Blue"]:
            make_light_slider(name)

        # Spot Angle
        self.lighting_controls["SpotAngle"] = QtWidgets.QDoubleSpinBox()
        self.lighting_controls["SpotAngle"].setRange(0.0, 180.0)
        self.lighting_controls["SpotAngle"].setSingleStep(1.0)
        self.lighting_controls["SpotAngle"].valueChanged.connect(self.on_light_controls_changed)
        lg.addRow("Spot Angle", self.lighting_controls["SpotAngle"])

        # Penumbra (Exponent)
        self.lighting_controls["Penumbra"] = QtWidgets.QDoubleSpinBox()
        self.lighting_controls["Penumbra"].setRange(0.0, 64.0)
        self.lighting_controls["Penumbra"].setSingleStep(1.0)
        self.lighting_controls["Penumbra"].valueChanged.connect(self.on_light_controls_changed)
        lg.addRow("Penumbra", self.lighting_controls["Penumbra"])

        # Shadows (placeholder)
        self.lighting_controls["Shadows"] = QtWidgets.QCheckBox("Casts Shadows")
        self.lighting_controls["Shadows"].stateChanged.connect(self.on_light_controls_changed)
        lg.addRow(self.lighting_controls["Shadows"])

        self.lighting_tab.setLayout(lg)

    def setup_details_tab(self):
        # Content widget that holds the groups (placed inside a scroll area)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
    
        self.details_labels = {}
    
        # Scene totals group
        scene_group = QtWidgets.QGroupBox("Scene Totals")
        scene_form = QtWidgets.QFormLayout()
        self.details_labels["SceneObjects"] = QtWidgets.QLabel("0 / 0")
        self.details_labels["SceneVerts"] = QtWidgets.QLabel("0")
        self.details_labels["SceneEdges"] = QtWidgets.QLabel("0")
        self.details_labels["SceneFaces"] = QtWidgets.QLabel("0")
        self.details_labels["SceneTris"] = QtWidgets.QLabel("0")
        self.details_labels["SceneMemory"] = QtWidgets.QLabel("N/A")
        scene_form.addRow("Objects (total / selected)", self.details_labels["SceneObjects"])
        scene_form.addRow("Verts", self.details_labels["SceneVerts"])
        scene_form.addRow("Edges", self.details_labels["SceneEdges"])
        scene_form.addRow("Faces", self.details_labels["SceneFaces"])
        scene_form.addRow("Tris", self.details_labels["SceneTris"])
        scene_form.addRow("Memory", self.details_labels["SceneMemory"])
        scene_group.setLayout(scene_form)
    
        # Active object group
        active_group = QtWidgets.QGroupBox("Active Object")
        active_form = QtWidgets.QFormLayout()
        self.details_labels["ObjType"] = QtWidgets.QLabel("N/A")
        self.details_labels["ObjName"] = QtWidgets.QLabel("N/A")
        self.details_labels["DimX"] = QtWidgets.QLabel("0.0")
        self.details_labels["DimY"] = QtWidgets.QLabel("0.0")
        self.details_labels["DimZ"] = QtWidgets.QLabel("0.0")
        self.details_labels["MeshVerts"] = QtWidgets.QLabel("0")
        self.details_labels["MeshEdges"] = QtWidgets.QLabel("0")
        self.details_labels["MeshFaces"] = QtWidgets.QLabel("0")
        self.details_labels["MeshTris"] = QtWidgets.QLabel("0")
        self.details_labels["Materials"] = QtWidgets.QLabel("N/A")
        self.details_labels["UVMaps"] = QtWidgets.QLabel("No")
        self.details_labels["ColorAttrCount"] = QtWidgets.QLabel("0")
        active_form.addRow("Type", self.details_labels["ObjType"])
        active_form.addRow("Name", self.details_labels["ObjName"])
        active_form.addRow("Dimensions X", self.details_labels["DimX"])
        active_form.addRow("Dimensions Y", self.details_labels["DimY"])
        active_form.addRow("Dimensions Z", self.details_labels["DimZ"])
        active_form.addRow("Mesh Verts", self.details_labels["MeshVerts"])
        active_form.addRow("Mesh Edges", self.details_labels["MeshEdges"])
        active_form.addRow("Mesh Faces", self.details_labels["MeshFaces"])
        active_form.addRow("Mesh Tris", self.details_labels["MeshTris"])
        active_form.addRow("Materials", self.details_labels["Materials"])
        active_form.addRow("UV Map", self.details_labels["UVMaps"])
        active_form.addRow("Color Attributes", self.details_labels["ColorAttrCount"])
        active_group.setLayout(active_form)
    
        # Edit/Selection group
        edit_group = QtWidgets.QGroupBox("Edit/Selection")
        edit_form = QtWidgets.QFormLayout()
        self.details_labels["SelVerts"] = QtWidgets.QLabel("0")
        self.details_labels["SelEdges"] = QtWidgets.QLabel("0")
        self.details_labels["SelFaces"] = QtWidgets.QLabel("0")
        edit_form.addRow("Selected Verts", self.details_labels["SelVerts"])
        edit_form.addRow("Selected Edges", self.details_labels["SelEdges"])
        edit_form.addRow("Selected Faces", self.details_labels["SelFaces"])
        edit_group.setLayout(edit_form)
    
        # (Removed duplicate light controls here; Lighting tab owns them.)
    
        layout.addWidget(scene_group)
        layout.addWidget(active_group)
        layout.addWidget(edit_group)
        layout.addStretch(1)
    
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(content)
    
        outer = QtWidgets.QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self.details_tab.setLayout(outer)

    def closeEvent(self, event):
        try:
            # Turn off any transform gizmos first
            try:
                if self.transform_widget:
                    self.transform_widget.Off()
                    self.transform_widget = None
            except Exception:
                pass
            # NEW: stop current tool if active
            try:
                if self.current_tool:
                    self.current_tool.stop(cancel=True)
                    self.current_tool = None
            except Exception:
                pass
            if self.camera_mode:
                self.exit_camera_mode()
            # Tell VTK side to stop rendering and detach observers
            if self.vtk_app:
                self.vtk_app.shutdown()

            app = QtWidgets.QApplication.instance()
            if hasattr(app, "_scene_windows"):
                app._scene_windows = [w for w in app._scene_windows if w is not self]
        finally:
            super().closeEvent(event)

    def on_create_subdivided_cube(self):
        """Ask user for subdivision count and add a subdivided cube to the scene."""
        n, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Subdivided Cube",
            "Subdivisions per edge:",
            3,      # default value
            1,      # minimum
            100,    # maximum
            1       # step
        )
        if not ok:
            return

        actor, base_name = self.vtk_app.create_subdivided_cube(n)
        if actor:
            # Use the helper so it gets a unique name and is added to the outliner
            self.add_actor_with_name(base_name, actor)


    def add_new_object(self, object_id, creation_func):
        """Create a new mesh object and add via undo stack (avoids immediate duplicate add)."""
        actor, base_name = creation_func(object_id)
        if not actor:
            return
        # Use undo command; its redo() performs the actual registration + outliner update.
        self.undo_stack.push(AddActorCommand(self, base_name, actor))

    def _add_actor_internal(self, base_name: str, actor: vtk.vtkActor, unique_name: str = None) -> str:
        if not actor:
            return ""
        if unique_name is None:
            count = len([key for key in self.object_registry if key.startswith(base_name)]) + 1
            unique_name = f"{base_name}_{count}" if count > 1 else base_name
        # register
        self.object_registry[unique_name] = actor
        actor.SetUserTransform(None)
        self.vtk_app.add_actor_to_scene(actor)
    
        # Choose parent collection: currently selected collection, else default "Collection"
        parent = None
        if isinstance(self.scene_outliner, QtWidgets.QTreeWidget):
            sel = self.scene_outliner.selectedItems()
            if sel and sel[0].data(0, QtCore.Qt.UserRole) == "collection":
                parent = sel[0]
            else:
                parent = self.ensure_collection("Collection")
    
            item = self._make_object_item(unique_name, "mesh", parent)
            self.scene_outliner.setCurrentItem(item)
            self.on_outliner_selection_changed(item)
        else:
            # Fallback (older list UI)
            list_item = QtWidgets.QListWidgetItem(unique_name, self.scene_outliner)
            list_item.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon))
            self.scene_outliner.setCurrentItem(list_item)
            self.on_outliner_selection_changed(list_item)
    
        self.update_scene_totals()
        return unique_name

    def _add_actor_no_undo(self, base_name: str, actor: vtk.vtkActor) -> str:
        return self._add_actor_internal(base_name, actor, unique_name=None)

    def _add_actor_no_undo_with_name(self, unique_name: str, actor: vtk.vtkActor) -> str:
        # Use provided name
        # Remove any existing duplicate
        if unique_name in self.object_registry:
            self._remove_object_silent(unique_name)
        return self._add_actor_internal(unique_name, actor, unique_name=unique_name)

    def _remove_object_silent(self, name: str):
        """Remove mesh object by name without prompts or status messages."""
        actor = self.object_registry.get(name)
        if not actor:
            return
        try:
            self.vtk_app.renderer.RemoveActor(actor)
            if actor in self.vtk_app.actors:
                self.vtk_app.actors.remove(actor)
        except Exception:
            pass
        self.object_registry.pop(name, None)

        # Remove from the outliner (tree-aware)
        if isinstance(self.scene_outliner, QtWidgets.QTreeWidget):
            it = self._find_tree_item_by_name(name)
            if it:
                parent = it.parent()
                if parent:
                    parent.removeChild(it)
                else:
                    idx = self.scene_outliner.indexOfTopLevelItem(it)
                    self.scene_outliner.takeTopLevelItem(idx)
        else:
            for i in range(self.scene_outliner.count()):
                if self.scene_outliner.item(i).text() == name:
                    self.scene_outliner.takeItem(i)
                    break

        if self.transform_widget and self.current_selected_actor is actor:
            try:
                self.transform_widget.Off()
            except Exception:
                pass
            self.transform_widget = None

        self.update_properties_panel(None)
        self.update_scene_totals()
        self.vtk_app.render_all()

    def _get_actor_user_matrix16(self, actor: vtk.vtkActor):
        """Return 16-list of UserTransform matrix, or identity if none."""
        m = vtk.vtkMatrix4x4()
        tf = actor.GetUserTransform()
        if tf:
            m.DeepCopy(tf.GetMatrix())
        else:
            m.Identity()
        return [m.GetElement(r, c) for r in range(4) for c in range(4)]

    def _apply_user_matrix16(self, actor: vtk.vtkActor, m16):
        m = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                m.SetElement(r, c, float(m16[r*4 + c]))
        tf = vtk.vtkTransform()
        tf.SetMatrix(m)
        actor.SetUserTransform(tf)
        # refresh UI
        self.update_properties_panel(actor)
        self.vtk_app.render_all()

    def record_transform_begin(self, actor: vtk.vtkActor):
        light, _ = self.get_selected_light()
        if light:
            return
        self._pending_transform = {
            "actor": actor,
            "before": self._get_actor_user_matrix16(actor)
        }

    def record_transform_end(self, actor: vtk.vtkActor):
        if not self._pending_transform or self._pending_transform.get("actor") is not actor:
            return
        after = self._get_actor_user_matrix16(actor)
        before = self._pending_transform["before"]
        if after != before:
            self.undo_stack.push(TransformActorCommand(self, actor, before, after))
        self._pending_transform = None

    def clear_scene(self):
        self.toggle_gizmo(False)
        # Ensure any transform widget is fully disabled
        if self.transform_widget:
            self.transform_widget.Off()
            self.transform_widget = None
        # NEW: stop tool if running
        if self.current_tool:
            self.current_tool.stop(cancel=True)
            self.current_tool = None
        
        if hasattr(self, "vertex_edit_action"):
            self.vertex_edit_action.setChecked(False)

        self.vtk_app.clear_scene()
        self.scene_outliner.clear()
        self.object_registry.clear()
        self.light_registry.clear()
        self.update_properties_panel(None)
        # Update scene totals after clear
        self.update_scene_totals()

    def get_selected_actor(self):
        """Return the selected mesh actor or light gizmo (supports QTreeWidget)."""
        try:
            items = self.scene_outliner.selectedItems()
        except Exception:
            return None
        if not items:
            return None

        if isinstance(self.scene_outliner, QtWidgets.QTreeWidget):
            # pick first non-collection
            chosen = None
            for it in items:
                if it.data(0, QtCore.Qt.UserRole) in ("mesh", "light"):
                    chosen = it
                    break
            if not chosen:
                return None
            name = chosen.text(0)
        else:
            name = items[0].text()

        if name in self.object_registry:
            return self.object_registry[name]
        if name in self.light_registry:
            return self.light_registry[name]['gizmo']
        return None
    
    def get_selected_light(self):
        """Return (vtkLight, name) for the selected light, else (None, None)."""
        try:
            items = self.scene_outliner.selectedItems()
        except Exception:
            return None, None
        if not items:
            return None, None

        if isinstance(self.scene_outliner, QtWidgets.QTreeWidget):
            # find first light item
            it = None
            for cand in items:
                if cand.data(0, QtCore.Qt.UserRole) == "light":
                    it = cand
                    break
            if not it:
                return None, None
            name = it.text(0)
        else:
            name = items[0].text()

        if name in self.light_registry:
            return self.light_registry[name]['light'], name
        return None, None

    def on_outliner_selection_changed(self, item):
        actor = self.get_selected_actor()
        self.update_properties_panel(actor)
        self.setup_transform_widget(actor) # Changed from setup_box_widget
        # Update scene totals to reflect new selection count
        self.update_scene_totals()

    def on_copy_selected(self):
        """Copy current selection (mesh or light) into internal clipboard."""
        sel_items = self.scene_outliner.selectedItems()
        if not sel_items:
            self.statusBar().showMessage("Copy: nothing selected")
            return

        # QTreeWidgetItem requires column index
        item = sel_items[0]
        name = item.text(0) if isinstance(item, QtWidgets.QTreeWidgetItem) else item.text()

        # Light?
        light, lname = self.get_selected_light()
        if light:
            entry = self.light_registry[lname]
            self.clipboard = {
                "kind": "light",
                "type": entry["type"],
                "position": light.GetPosition(),
                "focal": light.GetFocalPoint(),
                "intensity": light.GetIntensity(),
                "cone": light.GetConeAngle(),
                "exponent": light.GetExponent()
            }
            self.statusBar().showMessage(f'Copied light "{lname}"')
            return

        # Mesh?
        actor = self.object_registry.get(name)
        if actor:
            self.clipboard = {"kind": "mesh", "actor": actor, "name": name}
            self.statusBar().showMessage(f'Copied "{name}"')
        else:
            self.statusBar().showMessage("Copy: unsupported selection")


    def on_paste_selected(self):
        """Paste (duplicate) the last copied item."""
        if not self.clipboard:
            self.statusBar().showMessage("Paste: clipboard empty")
            return
        if self.clipboard.get("kind") == "mesh":
            new_name = self._duplicate_actor(self.clipboard["actor"], base_name=self.clipboard.get("name", "object"))
            self.statusBar().showMessage(f'Duplicated to "{new_name}"')
        elif self.clipboard.get("kind") == "light":
            self._duplicate_light_from_clipboard(self.clipboard)
        else:
            self.statusBar().showMessage("Paste: unsupported clipboard")
            

    def _duplicate_actor(self, src_actor: vtk.vtkActor, base_name: str = "object") -> str:
        """Create a duplicated mesh actor (deep-copied geometry and properties)."""
        if not src_actor or not src_actor.GetMapper():
            return ""
        # Deep-copy the source polydata
        data = src_actor.GetMapper().GetInput()
        poly = self.as_polydata(data)
        if not poly:
            return ""
        poly_copy = vtk.vtkPolyData()
        poly_copy.DeepCopy(poly)

        # Wrap in a trivial producer so we can reuse the same normals pipeline
        tp = vtk.vtkTrivialProducer()
        tp.SetOutput(poly_copy)

        mapper2 = self.vtk_app.create_mapper(tp)
        actor2 = self.vtk_app.create_actor(mapper2)

        # Copy visual properties
        actor2.GetProperty().DeepCopy(src_actor.GetProperty())

        # Copy actor transforms
        actor2.SetPosition(src_actor.GetPosition())
        actor2.SetOrientation(src_actor.GetOrientation())
        actor2.SetScale(src_actor.GetScale())
        if src_actor.GetUserTransform():
            tf = vtk.vtkTransform()
            tf.DeepCopy(src_actor.GetUserTransform())
            actor2.SetUserTransform(tf)

        # Small offset so duplicate is visible
        px, py, pz = actor2.GetPosition()
        actor2.SetPosition(px + 1.0, py + 1.0, pz)

        # Register and select
        dup_base = f"{base_name}_copy"
        self.add_actor_with_name(dup_base, actor2)
        return dup_base

    def _duplicate_light_from_clipboard(self, clip: dict):
        """Duplicate a light from clipboard settings."""
        ltype = clip.get("type", "Point")
        # Create via existing plumbing
        self.add_new_light(ltype)
        # Get the just-added light name (last item in outliner)
        item = self.scene_outliner.currentItem()
        if not item or item.text() not in self.light_registry:
            self.statusBar().showMessage("Paste: failed to add light")
            return
        name = item.text()
        entry = self.light_registry[name]
        light = entry["light"]
        # Apply copied settings + small offset
        pos = clip.get("position", (10, -10, 10))
        pos = (pos[0] + 1.0, pos[1] + 1.0, pos[2])
        light.SetPosition(*pos)
        light.SetFocalPoint(*clip.get("focal", (0, 0, 0)))
        light.SetIntensity(clip.get("intensity", 1.0))
        light.SetConeAngle(clip.get("cone", 180.0))
        light.SetExponent(clip.get("exponent", 1.0))
        # Move gizmo too
        entry["gizmo"].SetPosition(*pos)
        self.vtk_app.render_all()
        self.statusBar().showMessage(f'Duplicated light as "{name}"')

    def update_properties_panel(self, actor):
        self.block_signals = True
    
        # Light selected?
        light, light_name = self.get_selected_light()
        if light:
            # Tabs
            self.tabs.setTabEnabled(self.tabs.indexOf(self.lighting_tab), True)
            self.tabs.setTabEnabled(self.tabs.indexOf(self.appearance_tab), False)
    
            # Fill lighting UI
            self._populate_light_controls(light, self.light_registry[light_name]['type'])
    
            # Transform tab: only position is relevant for lights
            pos = light.GetPosition()
            for i, ax in enumerate("XYZ"):
                self.spinboxes[f"Position{ax}"].setValue(pos[i])
                self.spinboxes[f"Rotation{ax}"].setValue(0.0)
                self.spinboxes[f"Scale{ax}"].setValue(1.0)
    
            # Texture controls disabled for lights
            if getattr(self, "tex_load_button", None):
                self.tex_load_button.setEnabled(False)
            if getattr(self, "tex_clear_button", None):
                self.tex_clear_button.setEnabled(False)
            if getattr(self, "tex_thumb_label", None):
                self.tex_thumb_label.setPixmap(QtGui.QPixmap())
                self.tex_thumb_label.setToolTip("No texture")
    
            self.statusBar().showMessage(f"Selected Light: {light_name}")
            self.update_scene_totals()
            self.block_signals = False
            return
    
        # Mesh branch
        self.tabs.setTabEnabled(self.tabs.indexOf(self.lighting_tab), False)
        self.tabs.setTabEnabled(self.tabs.indexOf(self.appearance_tab), True)
    
        if actor:
            # Status
            cur_item = self.scene_outliner.currentItem()
            if cur_item:
                item_name = cur_item.text(0) if isinstance(cur_item, QtWidgets.QTreeWidgetItem) else cur_item.text()
                self.statusBar().showMessage(f"Selected: {item_name}")
            else:
                self.statusBar().showMessage("Selected: (unknown)")
    
            # Transform sync (UserTransform drives our UI; actor pos/orient/scale unused)
            transform = vtk.vtkTransform()
            if actor.GetUserTransform():
                transform.DeepCopy(actor.GetUserTransform())
    
            pos = transform.GetPosition()
            rot = transform.GetOrientation()
            scale = transform.GetScale()
    
            for i, axis in enumerate("XYZ"):
                self.spinboxes[f"Position{axis}"].setValue(pos[i])
                self.spinboxes[f"Rotation{axis}"].setValue(rot[i])
                self.spinboxes[f"Scale{axis}"].setValue(scale[i])
    
            # Appearance sync
            prop = actor.GetProperty()
            if prop:
                # Sliders
                self.sliders["Opacity"].setValue(int(prop.GetOpacity() * 100))
                self.sliders["Ambient"].setValue(int(prop.GetAmbient() * 100))
                self.sliders["Diffuse"].setValue(int(prop.GetDiffuse() * 100))
                self.sliders["Specular"].setValue(int(prop.GetSpecular() * 100))
                self.sliders["SpecularPower"].setValue(int(prop.GetSpecularPower()))
                self.combos["Interpolation"].setCurrentIndex(prop.GetInterpolation())
    
                # Representation (VTK: 2 Surface, 1 Wireframe, 0 Points)
                try:
                    rep = prop.GetRepresentation()
                    rep_idx = 0 if rep == 2 else (1 if rep == 1 else 2)  # 0 Surface, 1 Wireframe, 2 Points
                    if "Representation" in self.combos:
                        self.combos["Representation"].setCurrentIndex(rep_idx)
                except Exception:
                    pass
    
                # Edge/culling
                if "ShowEdges" in getattr(self, "checks", {}):
                    self.checks["ShowEdges"].setChecked(bool(prop.GetEdgeVisibility()))
                if "BackfaceCulling" in getattr(self, "checks", {}):
                    try:
                        self.checks["BackfaceCulling"].setChecked(bool(prop.GetBackfaceCulling()))
                    except Exception:
                        self.checks["BackfaceCulling"].setChecked(False)
                if "FrontfaceCulling" in getattr(self, "checks", {}):
                    try:
                        self.checks["FrontfaceCulling"].setChecked(bool(prop.GetFrontfaceCulling()))
                    except Exception:
                        self.checks["FrontfaceCulling"].setChecked(False)
    
            # Texture UI state
            if getattr(self, "tex_load_button", None):
                self.tex_load_button.setEnabled(True)
            if getattr(self, "tex_clear_button", None):
                has_tex = bool(actor.GetTexture())
                # If we have a remembered path, treat as textured for Clear button
                if not has_tex and actor in self.actor_texture_paths:
                    path = self.actor_texture_paths.get(actor)
                    has_tex = bool(path and os.path.exists(path))
                self.tex_clear_button.setEnabled(has_tex)
            self.update_texture_thumbnail(actor)
    
            # Details tab
            info = self.get_active_object_info(actor)
            self.details_labels["ObjType"].setText(info.get("type", "N/A"))
            self.details_labels["ObjName"].setText(info.get("name", "N/A"))
            self.details_labels["DimX"].setText(f'{info.get("dim_x", 0.0):.3f}')
            self.details_labels["DimY"].setText(f'{info.get("dim_y", 0.0):.3f}')
            self.details_labels["DimZ"].setText(f'{info.get("dim_z", 0.0):.3f}')
            self.details_labels["MeshVerts"].setText(str(info.get("verts, 0".split(",")[0]) if False else info.get("verts", 0)))
            self.details_labels["MeshEdges"].setText(str(info.get("edges", 0)))
            self.details_labels["MeshFaces"].setText(str(info.get("faces", 0)))
            self.details_labels["MeshTris"].setText(str(info.get("tris", 0)))
            self.details_labels["Materials"].setText("N/A")
            self.details_labels["UVMaps"].setText("Yes" if info.get("has_uv", False) else "No")
            self.details_labels["ColorAttrCount"].setText(str(info.get("color_attrs", 0)))
        else:
            # Nothing selected
            self.statusBar().showMessage("No object selected.")
            for widget in [self.spinboxes, self.sliders]:
                for w in widget.values():
                    try:
                        w.clearFocus()
                    except Exception:
                        pass
    
            # Disable tabs and reset defaults
            self.tabs.setTabEnabled(self.tabs.indexOf(self.lighting_tab), False)
            self.tabs.setTabEnabled(self.tabs.indexOf(self.appearance_tab), False)
            self.combos["Interpolation"].setCurrentIndex(0)
            if "Representation" in self.combos:
                self.combos["Representation"].setCurrentIndex(0)
            if "ShowEdges" in getattr(self, "checks", {}):
                self.checks["ShowEdges"].setChecked(False)
            if "BackfaceCulling" in getattr(self, "checks", {}):
                self.checks["BackfaceCulling"].setChecked(False)
            if "FrontfaceCulling" in getattr(self, "checks", {}):
                self.checks["FrontfaceCulling"].setChecked(False)
    
            # Texture UI disabled and cleared
            if getattr(self, "tex_load_button", None):
                self.tex_load_button.setEnabled(False)
            if getattr(self, "tex_clear_button", None):
                self.tex_clear_button.setEnabled(False)
            if getattr(self, "tex_thumb_label", None):
                self.tex_thumb_label.setPixmap(QtGui.QPixmap())
                self.tex_thumb_label.setToolTip("No texture")
    
            # Clear details labels
            self.details_labels["ObjType"].setText("N/A")
            self.details_labels["ObjName"].setText("N/A")
            self.details_labels["DimX"].setText("0.0")
            self.details_labels["DimY"].setText("0.0")
            self.details_labels["DimZ"].setText("0.0")
            self.details_labels["MeshVerts"].setText("0")
            self.details_labels["MeshEdges"].setText("0")
            self.details_labels["MeshFaces"].setText("0")
            self.details_labels["MeshTris"].setText("0")
            self.details_labels["Materials"].setText("N/A")
            self.details_labels["UVMaps"].setText("No")
            self.details_labels["ColorAttrCount"].setText("0")
    
        # Always refresh scene totals
        self.update_scene_totals()
        self.block_signals = False

    def _set_appearance_controls_enabled(self, enabled: bool):
        try:
            for s in self.sliders.values():
                s.setEnabled(enabled)
            for c in self.combos.values():
                c.setEnabled(enabled)
            for chk in getattr(self, "checks", {}).values():
                chk.setEnabled(enabled)
        except Exception:
            pass

    # -------------------------- Texture ------------------------

    def eventFilter(self, obj, event):
        # Clickable thumbnail behaviour: Left=Load…, Right=Show in Explorer
        if getattr(self, "tex_thumb_label", None) is obj and event.type() == QtCore.QEvent.MouseButtonPress:
            btn = event.button()
            if btn == QtCore.Qt.LeftButton:
                self.on_load_texture_clicked()
            elif btn == QtCore.Qt.RightButton:
                actor = self.get_selected_actor()
                if not actor or self.get_selected_light()[0]:
                    return False
                path = self.actor_texture_paths.get(actor)
                if path and os.path.exists(path):
                    try:
                        # Windows Explorer highlight
                        os.startfile(os.path.dirname(path))
                    except Exception:
                        pass
            return True
        return super().eventFilter(obj, event)

    def on_load_texture_clicked(self):
        light, _ = self.get_selected_light()
        actor = self.get_selected_actor()
        if not actor or light:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Color Texture", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path:
            return
        # Single, centralized path (adds UVs, sets color to white, updates UI)
        self._apply_texture(actor, path)

    def on_clear_texture_clicked(self):
        light, _ = self.get_selected_light()
        actor = self.get_selected_actor()
        if not actor or light:
            return
        actor.SetTexture(None)
        if actor in self.actor_texture_paths:
            self.actor_texture_paths.pop(actor, None)
        self.update_texture_thumbnail(actor)
        if getattr(self, "tex_clear_button", None):
            self.tex_clear_button.setEnabled(False)
        self.vtk_app.render_all()

    def _apply_texture(self, actor: vtk.vtkActor, path: str):
        """Load texture, ensure UVs, apply, update UI."""
        tex = self.vtk_app.load_texture(path)
        if not tex:
            QtWidgets.QMessageBox.warning(self, "Load Texture", "Could not load the selected image.")
            return
        # Ensure UVs first
        self._ensure_texture_coordinates(actor)
        actor.SetTexture(tex)
        # Neutral base color (avoid tinting)
        actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        self.actor_texture_paths[actor] = path
        self.update_texture_thumbnail(actor)
        if getattr(self, "tex_clear_button", None):
            self.tex_clear_button.setEnabled(True)
        self.vtk_app.render_all()

    def _ensure_texture_coordinates(self, actor: vtk.vtkActor):
        """Generate texture coordinates if the mesh has none."""
        if not actor or not actor.GetMapper() or not actor.GetMapper().GetInput():
            return
        poly = actor.GetMapper().GetInput()
        if poly.GetPointData().GetTCoords():
            return  # already has UVs

        # Choose a simple mapping based on rough shape (name heuristic)
        name = ""
        cur_item = self.scene_outliner.currentItem()
        if cur_item:
            name = cur_item.text(0).lower()

        src_pd = poly

        # Sphere-like
        if "sphere" in name:
            mapper_filter = vtk.vtkTextureMapToSphere()
            mapper_filter.SetPreventSeam(1)
            mapper_filter.SetInputData(src_pd)
        # Cylinder / cone
        elif "cyl" in name or "cone" in name:
            mapper_filter = vtk.vtkTextureMapToCylinder()
            mapper_filter.PreventSeamOn()
            mapper_filter.SetInputData(src_pd)
        # Box/cube: map each face once (0..1 per face)
        elif "cube" in name or "box" in name:
            mapper_filter = vtk.vtkTextureMapToBox()
            mapper_filter.AutomaticBoxGenerationOn()
            mapper_filter.PreventSeamOn()
            mapper_filter.SetInputData(src_pd)
        # Plane / rectangle
        elif "rect" in name or "plane" in name or "pyramid" in name:
            mapper_filter = vtk.vtkTextureMapToPlane()
            mapper_filter.AutomaticPlaneGenerationOn()
            mapper_filter.SetInputData(src_pd)
        else:
            # Generic planar projection
            mapper_filter = vtk.vtkTextureMapToPlane()
            mapper_filter.AutomaticPlaneGenerationOn()
            mapper_filter.SetInputData(src_pd)

        mapper_filter.Update()

        # Rebuild lighting pipeline (normals etc.) on UV-mapped output
        tp = vtk.vtkTrivialProducer()
        tp.SetOutput(mapper_filter.GetOutput())
        new_mapper = self.vtk_app.create_mapper(tp)

        # Preserve property + texture (if any)
        old_prop = vtk.vtkProperty()
        old_prop.DeepCopy(actor.GetProperty())
        old_tex = actor.GetTexture()

        actor.SetMapper(new_mapper)
        actor.GetProperty().DeepCopy(old_prop)
        if old_tex:
            actor.SetTexture(old_tex)

    def update_texture_thumbnail(self, actor=None):
        # Update thumbnail button preview (small square)
        if not getattr(self, "tex_thumb_label", None):
            return
        if actor is None:
            actor = self.get_selected_actor()
        # default empty look
        self.tex_thumb_label.setPixmap(QtGui.QPixmap())
        self.tex_thumb_label.setToolTip("No texture")
        # show image if known path
        path = self.actor_texture_paths.get(actor)
        if path and os.path.exists(path):
            pm = QtGui.QPixmap(path)
            if not pm.isNull():
                self.tex_thumb_label.setPixmap(pm.scaled(
                    self.tex_thumb_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
                self.tex_thumb_label.setToolTip(path)

    def as_polydata(self, data_obj):
        """Return vtkPolyData from any VTK dataset (or None if not convertible)."""
        if data_obj is None:
            return None
        if isinstance(data_obj, vtk.vtkPolyData):
            return data_obj
        try:
            gf = vtk.vtkGeometryFilter()
            gf.SetInputData(data_obj)
            gf.Update()
            return gf.GetOutput()
        except Exception:
            return None

    def compute_polydata_stats(self, poly):
        """Compute verts/edges/faces/tris and memory for a vtkPolyData."""
        stats = {"verts": 0, "edges": 0, "faces": 0, "tris": 0, "memory_kb": 0}
        if poly is None:
            return stats

        # Ensure pipeline is updated
        poly.Modified()
        poly.BuildCells()

        stats["verts"] = poly.GetNumberOfPoints()
        # Faces: polygons + triangle strips as polygonal faces
        stats["faces"] = poly.GetNumberOfPolys() + poly.GetNumberOfStrips()

        # Edges via extract edges (unique topological edges)
        ext = vtk.vtkExtractEdges()
        ext.SetInputData(poly)
        ext.Update()
        edges_out = ext.GetOutput()
        stats["edges"] = edges_out.GetNumberOfLines()

        # Triangles via triangulation
        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(poly)
        tri.Update()
        tri_out = tri.GetOutput()
        stats["tris"] = tri_out.GetNumberOfPolys()

        # Memory (KB) from polydata
        stats["memory_kb"] = poly.GetActualMemorySize()
        return stats

    def compute_scene_totals(self):
        """Compute scene-wide totals across all objects in the outliner/registry."""
        totals = {"verts": 0, "edges": 0, "faces": 0, "tris": 0, "memory_kb": 0}
        for name, actor in self.object_registry.items():
            mapper = actor.GetMapper()
            if not mapper:
                continue
            data = mapper.GetInput()
            poly = self.as_polydata(data)
            s = self.compute_polydata_stats(poly)
            totals["verts"] += s["verts"]
            totals["edges"] += s["edges"]
            totals["faces"] += s["faces"]
            totals["tris"] += s["tris"]
            totals["memory_kb"] += s["memory_kb"]
        return totals

    def update_scene_totals(self):
        """Update the Scene Totals labels."""
        totals = self.compute_scene_totals()
        total_objects = len(self.object_registry)
        selected_objects = len(self.scene_outliner.selectedItems())

        self.details_labels["SceneObjects"].setText(f"{total_objects} / {selected_objects}")
        self.details_labels["SceneVerts"].setText(str(totals["verts"]))
        self.details_labels["SceneEdges"].setText(str(totals["edges"]))
        self.details_labels["SceneFaces"].setText(str(totals["faces"]))
        self.details_labels["SceneTris"].setText(str(totals["tris"]))

        mb = totals["memory_kb"] / 1024.0 if totals["memory_kb"] else 0.0
        self.details_labels["SceneMemory"].setText(f"{mb:.2f} MB" if mb > 0 else "N/A")

    def count_color_arrays(self, pd):
        """Count color-like attributes in point/cell data."""
        def arrays_in_data(data):
            count = 0
            n = data.GetNumberOfArrays()
            for i in range(n):
                arr = data.GetArray(i)
                if arr is None:
                    continue
                name = arr.GetName() or ""
                name_l = name.lower()
                # Heuristics: name contains 'color' OR is uchar with 3/4 components.
                is_color_name = "color" in name_l
                is_color_type = arr.IsA("vtkUnsignedCharArray") and arr.GetNumberOfComponents() in (3, 4)
                if is_color_name or is_color_type:
                    count += 1
            return count

        if pd is None:
            return 0
        return arrays_in_data(pd.GetPointData()) + arrays_in_data(pd.GetCellData())

    def get_active_object_info(self, actor):
        """Collect active object info and mesh stats."""
        info = {
            "type": "N/A",
            "name": "N/A",
            "dim_x": 0.0,
            "dim_y": 0.0,
            "dim_z": 0.0,
            "verts": 0,
            "edges": 0,
            "faces": 0,
            "tris": 0,
            "has_uv": False,
            "color_attrs": 0,
        }

        if actor is None:
            return info

        # ---- Name / type ----
        cur_item = self.scene_outliner.currentItem()
        if cur_item:
            info["name"] = cur_item.text(0)

        # Decide basic type
        if actor in self.object_registry.values():
            info["type"] = "Mesh"
        else:
            # Just in case this ever gets called for a light gizmo
            for name, entry in self.light_registry.items():
                if entry.get("gizmo") is actor:
                    info["type"] = "Light Gizmo"
                    info["name"] = name
                    return info

        # ---- Dimensions from actor bounds (world-space) ----
        try:
            bounds = actor.GetBounds()
        except Exception:
            bounds = None

        if bounds is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = bounds
            if xmax > xmin and ymax > ymin and zmax > zmin:
                info["dim_x"] = float(xmax - xmin)
                info["dim_y"] = float(ymax - ymin)
                info["dim_z"] = float(zmax - zmin)

        # ---- Mesh statistics (verts / edges / faces / tris / UV / color attrs) ----
        mapper = actor.GetMapper()
        if mapper:
            data = mapper.GetInput()
            poly = self.as_polydata(data)
            if poly:
                stats = self.compute_polydata_stats(poly)
                info["verts"] = stats.get("verts", 0)
                info["edges"] = stats.get("edges", 0)
                info["faces"] = stats.get("faces", 0)
                info["tris"] = stats.get("tris", 0)

                # UV map?
                tcoords = poly.GetPointData().GetTCoords()
                info["has_uv"] = bool(tcoords and tcoords.GetNumberOfTuples() > 0)

                # Color attributes
                info["color_attrs"] = self.count_color_arrays(poly)

        return info

    def on_transform_changed(self):
        if self.block_signals:
            return

        # If a light is selected, never apply a UserTransform to its gizmo.
        light, lname = self.get_selected_light()
        if light:
            # Preserve current direction while changing position via spinboxes
            px0, py0, pz0 = light.GetPosition()
            fx, fy, fz = light.GetFocalPoint()
            dir_vec = (fx - px0, fy - py0, fz - pz0)

            pos = [self.spinboxes[f"Position{ax}"].value() for ax in "XYZ"]
            light.SetPosition(*pos)
            gizmo = self.light_registry[lname]['gizmo']
            gizmo.SetPosition(*pos)
            light.SetFocalPoint(pos[0] + dir_vec[0], pos[1] + dir_vec[1], pos[2] + dir_vec[2])

            # Recenter the current translate circle if present
            if self.transform_widget:
                try:
                    rep = self.transform_widget.GetRepresentation()
                    rep.PlaceWidget(gizmo.GetBounds())
                    rep.SetOrigin(gizmo.GetPosition())
                except Exception:
                    pass

            self._populate_light_controls(light, self.light_registry[lname]['type'])
            self.vtk_app.render_all()
            return

        # Mesh objects: original behavior
        actor = self.get_selected_actor()
        if actor:
            transform = vtk.vtkTransform()
            pos = [self.spinboxes[f"Position{ax}"].value() for ax in "XYZ"]
            rot = [self.spinboxes[f"Rotation{ax}"].value() for ax in "XYZ"]
            scale = [self.spinboxes[f"Scale{ax}"].value() for ax in "XYZ"]

            transform.Identity()
            transform.Translate(pos)
            transform.RotateZ(rot[2])
            transform.RotateY(rot[1])
            transform.RotateX(rot[0])
            transform.Scale(scale)

            actor.SetUserTransform(transform)

            if self.transform_widget:
                self.setup_transform_widget(actor)

            self.vtk_app.render_all()

    def on_appearance_changed(self):
        if self.block_signals: return
        actor = self.get_selected_actor()
        light, _ = self.get_selected_light()
        if not actor or light: return

        discrete = False
        sender = self.sender()
        if isinstance(sender, (QtWidgets.QComboBox, QtWidgets.QCheckBox)):
            discrete = True

        before = self._get_actor_property_snapshot(actor) if discrete else None

        prop = actor.GetProperty()
        prop.SetOpacity(self.sliders["Opacity"].value() / 100.0)
        prop.SetAmbient(self.sliders["Ambient"].value() / 100.0)
        prop.SetDiffuse(self.sliders["Diffuse"].value() / 100.0)
        prop.SetSpecular(self.sliders["Specular"].value() / 100.0)
        prop.SetSpecularPower(self.sliders["SpecularPower"].value())

        # Reconfigure the existing normals filter instead of stacking a new one
        interp_idx = self.combos["Interpolation"].currentIndex()
        mapper = actor.GetMapper()
        if mapper:
            normals = getattr(mapper, "_vt_normals", None)

            # If we didn't build this mapper, try to locate a vtkPolyDataNormals upstream
            if normals is None:
                try:
                    alg = mapper.GetInputConnection(0, 0).GetProducer() if mapper.GetInputConnection(0,0) else None
                    while alg:
                        if isinstance(alg, vtk.vtkPolyDataNormals):
                            normals = alg
                            break
                        alg = alg.GetInputConnection(0, 0).GetProducer() if alg.GetInputConnection(0,0) else None
                except Exception:
                    normals = None

            if normals:
                normals.AutoOrientNormalsOn()
                normals.ConsistencyOn()
                if interp_idx == 0:  # Flat
                    prop.SetInterpolationToFlat()
                    normals.ComputeCellNormalsOn()
                    normals.ComputePointNormalsOff()
                    normals.SplittingOff()
                elif interp_idx == 1:  # Gouraud
                    prop.SetInterpolationToGouraud()
                    normals.ComputeCellNormalsOff()
                    normals.ComputePointNormalsOn()
                    normals.SplittingOn()              # CHANGED
                    normals.SetFeatureAngle(30.0)      # ensure 90° cube edges are sharp
                elif interp_idx == 2:  # Phong
                    prop.SetInterpolationToPhong()
                    normals.ComputeCellNormalsOff()
                    normals.ComputePointNormalsOn()
                    normals.SplittingOn()              # CHANGED
                    normals.SetFeatureAngle(30.0)
                normals.Modified()
                try:
                    normals.Update()
                except Exception:
                    pass

        mode_idx = self.combos.get("Representation").currentIndex() if "Representation" in self.combos else 0
        if mode_idx == 0: prop.SetRepresentationToSurface()
        elif mode_idx == 1: prop.SetRepresentationToWireframe()
        else: prop.SetRepresentationToPoints()

        if self.checks.get("ShowEdges"): 
            prop.SetEdgeVisibility(self.checks["ShowEdges"].isChecked())
        if self.checks.get("BackfaceCulling"):
            prop.BackfaceCullingOn() if self.checks["BackfaceCulling"].isChecked() else prop.BackfaceCullingOff()
        if self.checks.get("FrontfaceCulling"):
            prop.FrontfaceCullingOn() if self.checks["FrontfaceCulling"].isChecked() else prop.FrontfaceCullingOff()

        self.slider_value_labels["Opacity"].setText(f"{self.sliders['Opacity'].value()}%")
        self.slider_value_labels["Ambient"].setText(f"{self.sliders['Ambient'].value()}%")
        self.slider_value_labels["Diffuse"].setText(f"{self.sliders['Diffuse'].value()}%")
        self.slider_value_labels["Specular"].setText(f"{self.sliders['Specular'].value()}%")
        self.slider_value_labels["SpecularPower"].setText(str(self.sliders["SpecularPower"].value()))

        self.vtk_app.render_all()

        if discrete:
            after = self._get_actor_property_snapshot(actor)
            if after != before:
                self.undo_stack.push(PropertyChangeCommand(self, actor, before, after))

    def change_current_object_color(self):
        """Change color for all selected mesh objects."""
        # Collect selected mesh actors (exclude lights/collections)
        selected_items = self.scene_outliner.selectedItems()
        actors = []
        for item in selected_items:
            if item.data(0, QtCore.Qt.UserRole) == "mesh":
                name = item.text(0)
                actor = self.object_registry.get(name)
                if actor:
                    actors.append(actor)
        
        if not actors:
            QtWidgets.QMessageBox.information(self, "Change Color", "No mesh objects selected.")
            return
        
        # Pick color once
        color = QtWidgets.QColorDialog.getColor()
        if not color.isValid():
            return
        
        rgb = (color.redF(), color.greenF(), color.blueF())
        
        # Batch undo: snapshot all, change all, push one command
        snapshots_before = []
        for actor in actors:
            snapshots_before.append((actor, self._get_actor_property_snapshot(actor)))
        
        for actor in actors:
            self.vtk_app.change_color(rgb, actor)
        
        snapshots_after = []
        for actor in actors:
            snapshots_after.append((actor, self._get_actor_property_snapshot(actor)))
        
        # Push a batch property command
        self.undo_stack.push(BatchPropertyChangeCommand(self, snapshots_before, snapshots_after))
        
        self.statusBar().showMessage(f"Changed color of {len(actors)} object(s)")

     # ===== Export helpers =====
    def polydata_from_actor(self, actor: vtk.vtkActor, apply_transform=True) -> vtk.vtkPolyData:
        """Return a (optionally world-transformed) vtkPolyData from an actor."""
        if not actor or not actor.GetMapper() or not actor.GetMapper().GetInput():
            return None
        data = actor.GetMapper().GetInput()
        poly = self.as_polydata(data)
        if poly is None:
            return None
        if apply_transform:
            mat = vtk.vtkMatrix4x4()
            actor.GetMatrix(mat)  # world transform (includes UserTransform)
            tf = vtk.vtkTransform()
            tf.SetMatrix(mat)
            tpf = vtk.vtkTransformPolyDataFilter()
            tpf.SetInputData(poly)
            tpf.SetTransform(tf)
            tpf.Update()
            poly = tpf.GetOutput()

        # Optional: clean before write
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(poly)
        cleaner.Update()
        return cleaner.GetOutput()

    def write_polydata(self, poly: vtk.vtkPolyData, filepath: str) -> bool:
        """Write polydata to disk based on file extension."""
        if poly is None or not filepath:
            return False
        ext = os.path.splitext(filepath)[1].lower()
        writer = None
        if ext == ".stl":
            writer = vtk.vtkSTLWriter(); writer.SetFileTypeToBinary()
        elif ext == ".obj":
            writer = vtk.vtkOBJWriter()
        elif ext == ".ply":
            writer = vtk.vtkPLYWriter(); writer.SetFileTypeToBinary()
        elif ext == ".vtp":
            writer = vtk.vtkXMLPolyDataWriter(); writer.SetDataModeToBinary()
        elif ext == ".vtk":
            writer = vtk.vtkPolyDataWriter(); writer.SetFileTypeToBinary()
        else:
            # Default to OBJ if unknown
            writer = vtk.vtkOBJWriter()
            filepath = filepath + ".obj"

        writer.SetFileName(filepath)
        writer.SetInputData(poly)
        ok = bool(writer.Write())
        return ok
    
    def _switch_edit_submode(self, which: str):
        """Switch between 'vertex' and 'face' edit tools while Edit Mode is ON."""
        if not self.vertex_edit_button.isChecked():
            return
        # normalize buttons
        if which == "vertex":
            self.edit_vertex_mode_btn.setChecked(True)
            self.edit_face_mode_btn.setChecked(False)
        else:
            self.edit_vertex_mode_btn.setChecked(False)
            self.edit_face_mode_btn.setChecked(True)

        # stop current tool
        if self.current_tool:
            try:
                self.current_tool.stop(cancel=False)
            except Exception:
                pass
            self.current_tool = None

        if which == "vertex":
            # Start vertex edit tool (existing)
            actor = self.get_selected_actor()
            light, _ = self.get_selected_light()
            if not actor or light:
                QtWidgets.QMessageBox.information(self, "Vertex Edit", "Select a mesh object first.")
                return
            self.current_tool = VertexEditTool(self)
            self.current_tool.start()
            self.statusBar().showMessage("Vertex Edit: click a vertex then drag to move it.")
        else:
            # Start face edit tool
            actor = self.get_selected_actor()
            light, _ = self.get_selected_light()
            if not actor or light or not self._is_face_edit_allowed(actor):
                QtWidgets.QMessageBox.information(self, "Face Edit", "Select a polygonal mesh object (e.g., SubdividedCube).")
                return
            self.current_tool = FaceEditTool(self)
            self.current_tool.start()
            self.statusBar().showMessage("Face Edit: click a face to select; drag to move along normal; press Delete (or button) to remove.")
    
    def _is_face_edit_allowed(self, actor: vtk.vtkActor) -> bool:
            """Allow only simple polygonal meshes (ignore volumes/implicit parametric)."""
            try:
                data = actor.GetMapper().GetInput()
                poly = self.as_polydata(data)
                if not isinstance(poly, vtk.vtkPolyData):
                    return False
                return poly.GetNumberOfPolys() > 0
            except Exception:
                return False

    def on_vertex_edit_toggled(self, checked):
        # Sync the action and button states
        self.block_signals = True
        if hasattr(self, "vertex_edit_action") and self.sender() is not self.vertex_edit_action:
            self.vertex_edit_action.setChecked(checked)
        if self.sender() is not self.vertex_edit_button:
            self.vertex_edit_button.setChecked(checked)
        self.block_signals = False

        # Enable/disable transform spinboxes
        for sb in self.spinboxes.values():
            sb.setEnabled(not checked)

        self.vertex_edit_button.setText("Exit Edit Mode" if checked else "Enter Edit Mode")

        # NEW: while in Edit Mode, hide gizmo and transform buttons overlay
        try:
            if checked:
                # turn off any active gizmo cleanly
                if self.transform_widget:
                    self.transform_widget.Off()
                    self.transform_widget = None
                # keep the action in sync (unchecked)
                if self.toggle_gizmo_action.isChecked():
                    self.toggle_gizmo_action.setChecked(False)
                # hide on‑screen transform mode buttons (move/rotate/scale)
                if hasattr(self, "transform_button_container") and self.transform_button_container:
                    self.transform_button_container.setVisible(False)
            else:
                # show overlay buttons again
                if hasattr(self, "transform_button_container") and self.transform_button_container:
                    self.transform_button_container.setVisible(True)
        except Exception:
            pass

        # Sub-mode buttons visibility
        if checked:
            self.edit_vertex_mode_btn.show()
            self.edit_face_mode_btn.show()
            # stop other tool, then default to Vertex submode
            if self.current_tool:
                try:
                    self.current_tool.stop(cancel=False)
                except Exception:
                    pass
                self.current_tool = None
            self.edit_vertex_mode_btn.setChecked(True)
            self.edit_face_mode_btn.setChecked(False)
            self._switch_edit_submode("vertex")
            self.statusBar().showMessage("Edit Mode: ON (Vertex)")
        else:
            self.edit_vertex_mode_btn.hide()
            self.edit_face_mode_btn.hide()
            if self.current_tool:
                try:
                    self.current_tool.stop(cancel=False)
                except Exception:
                    pass
                self.current_tool = None
            self.statusBar().showMessage("Edit Mode: OFF")

    def export_selected(self):
        """Export the currently selected mesh to a file (with transforms applied)."""
        # Ignore lights
        sel = self.scene_outliner.selectedItems()
        if not sel:
            QtWidgets.QMessageBox.information(self, "Export Selected", "No object selected.")
            return

        # Find first selected mesh item (skip collections/lights)
        item = None
        if isinstance(self.scene_outliner, QtWidgets.QTreeWidget):
            for it in sel:
                if it.data(0, QtCore.Qt.UserRole) == "mesh":
                    item = it
                    break
        else:
            item = sel[0]

        if not item:
            QtWidgets.QMessageBox.information(self, "Export Selected", "Please select a mesh object.")
            return

        name = item.text(0) if isinstance(item, QtWidgets.QTreeWidgetItem) else item.text()
        if name in self.light_registry:
            QtWidgets.QMessageBox.information(self, "Export Selected", "Selected item is a light. Select a mesh object.")
            return

        actor = self.object_registry.get(name)
        if not actor:
            QtWidgets.QMessageBox.information(self, "Export Selected", "Selected item is not a mesh.")
            return

        # Choose file
        filters = "Wavefront OBJ (*.obj);;STL (*.stl);;PLY (*.ply);;VTK PolyData (*.vtk);;VTK XML PolyData (*.vtp)"
        default_name = f"{name}.obj"
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Selected", default_name, filters)
        if not filepath:
            return

        poly = self.polydata_from_actor(actor, apply_transform=True)
        if poly is None:
            QtWidgets.QMessageBox.warning(self, "Export Selected", "Could not extract mesh for export.")
            return

        if self.write_polydata(poly, filepath):
            self.statusBar().showMessage(f'Exported "{name}" to {os.path.basename(filepath)}')
        else:
            QtWidgets.QMessageBox.critical(self, "Export Selected", "Failed to write file.")

    def export_all_to_directory(self):
        """Export all mesh objects to a chosen directory as separate files (same format)."""
        if not self.object_registry:
            QtWidgets.QMessageBox.information(self, "Export All", "Scene has no mesh objects.")
            return

        target_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Export Directory")
        if not target_dir:
            return

        # Choose format
        items = ["OBJ (*.obj)", "STL (*.stl)", "PLY (*.ply)", "VTP (*.vtp)", "Legacy VTK (*.vtk)"]
        item, ok = QtWidgets.QInputDialog.getItem(self, "Choose Format", "Format:", items, 0, False)
        if not ok:
            return
        ext_map = {
            "OBJ (*.obj)": ".obj",
            "STL (*.stl)": ".stl",
            "PLY (*.ply)": ".ply",
            "VTP (*.vtp)": ".vtp",
            "Legacy VTK (*.vtk)": ".vtk",
        }
        ext = ext_map[item]

        count_ok, count_fail = 0, 0
        for name, actor in self.object_registry.items():
            # sanitize filename
            base = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
            out_path = os.path.join(target_dir, base + ext)
            poly = self.polydata_from_actor(actor, apply_transform=True)
            if poly is None or not self.write_polydata(poly, out_path):
                count_fail += 1
            else:
                count_ok += 1

        self.statusBar().showMessage(f"Exported {count_ok} object(s), {count_fail} failed. Directory: {target_dir}")

    def export_scene_as_one(self):
        """Export all mesh objects merged into one mesh file."""
        if not self.object_registry:
            QtWidgets.QMessageBox.information(self, "Export Scene", "Scene has no mesh objects.")
            return

        filters = "Wavefront OBJ (*.obj);;STL (*.stl);;PLY (*.ply);;VTK PolyData (*.vtk);;VTK XML PolyData (*.vtp)"
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Scene As One Mesh", "scene.obj", filters)
        if not filepath:
            return

        append = vtk.vtkAppendPolyData()
        any_data = False
        for _, actor in self.object_registry.items():
            poly = self.polydata_from_actor(actor, apply_transform=True)
            if poly and poly.GetNumberOfPoints() > 0:
                append.AddInputData(poly)
                any_data = True

        if not any_data:
            QtWidgets.QMessageBox.warning(self, "Export Scene", "No valid mesh data to export.")
            return

        append.Update()
        # Clean and (optional) triangulate for formats like STL
        cleaned = vtk.vtkCleanPolyData()
        cleaned.SetInputData(append.GetOutput())
        cleaned.Update()
        out_poly = cleaned.GetOutput()

        # For STL, ensure triangles
        if os.path.splitext(filepath)[1].lower() == ".stl":
            tri = vtk.vtkTriangleFilter()
            tri.SetInputData(out_poly)
            tri.Update()
            out_poly = tri.GetOutput()

        if self.write_polydata(out_poly, filepath):
            self.statusBar().showMessage(f"Exported scene to {os.path.basename(filepath)}")
        else:
            QtWidgets.QMessageBox.critical(self, "Export Scene", "Failed to write file.")


    #------------  UI helper -------------------------------------
    
    def _repolish_ui(self):
        """Force Qt to re-apply palette/stylesheet to all widgets (prevents stale hover colors)."""
        app = QtWidgets.QApplication.instance()
        if not app:
            return
        # Recreate the style object (flush caches)
        try:
            current_style_name = app.style().objectName()
            app.setStyle(QStyleFactory.create(current_style_name))
        except Exception:
            pass
        for w in app.allWidgets():
            try:
                st = w.style()
                st.unpolish(w)
                st.polish(w)
                w.update()
            except Exception:
                pass
    
    # -------- NEW: Lighting controls plumbing --------
    def on_toggle_lighting(self, checked):
        """Enable/disable lighting globally."""
        self.lighting_enabled = checked
        self.toggle_lighting_action.setText("Lighting On" if checked else "Lighting Off")

        # Renderer side: keep two-sided lighting when enabled
        self.vtk_app.renderer.SetTwoSidedLighting(bool(checked))

        for actor in self.object_registry.values():
            if checked:
                actor.GetProperty().LightingOn()
            else:
                actor.GetProperty().LightingOff()
        self.vtk_app.render_all()

    def _populate_light_controls(self, light, light_type: str):
        """Populate the lighting UI with current light properties (RGB color)."""
        self.block_signals = True
        lc = self.lighting_controls
    
        # Set light type combo
        type_index = {"Point": 0, "Directional": 1, "Spot": 2}.get(light_type, 0)
        lc["Type"].setCurrentIndex(type_index)
    
        # Set position
        px, py, pz = light.GetPosition()
        lc["PosX"].setValue(px)
        lc["PosY"].setValue(py)
        lc["PosZ"].setValue(pz)
    
        # Set direction
        dir_vec = self._light_direction_from_light(light)
        lc["DirX"].setValue(dir_vec[0])
        lc["DirY"].setValue(dir_vec[1])
        lc["DirZ"].setValue(dir_vec[2])
    
        # Set intensity
        lc["Intensity"].setValue(light.GetIntensity())
    
        # FIXED: Use GetDiffuseColor() instead of GetColor()
        r, g, b = light.GetDiffuseColor()
        lc["Red"].setValue(int(r * 100))
        lc["Green"].setValue(int(g * 100))
        lc["Blue"].setValue(int(b * 100))
        self.lighting_slider_value_labels["Red"].setText(f"{int(r*100)}%")
        self.lighting_slider_value_labels["Green"].setText(f"{int(g*100)}%")
        self.lighting_slider_value_labels["Blue"].setText(f"{int(b*100)}%")
    
        # Enable/disable spot controls
        lc["SpotAngle"].setEnabled(light_type == "Spot")
        lc["Penumbra"].setEnabled(light_type == "Spot")
        lc["SpotAngle"].setValue(light.GetConeAngle())
        lc["Penumbra"].setValue(light.GetExponent())
    
        # Shadows (placeholder, always unchecked)
        lc["Shadows"].setChecked(False)
    
        self.block_signals = False

    def on_light_controls_changed(self, *args):
        """Handle light control changes with RGB color support."""
        if self.block_signals:
            return
        light, lname = self.get_selected_light()
        if not light:
            return
    
        lc = self.lighting_controls
    
        # Update light type
        idx = lc["Type"].currentIndex()
        type_map = {0: "Point", 1: "Directional", 2: "Spot"}
        sel_type = type_map.get(idx, "Point")
        if sel_type != self.light_registry[lname]['type']:
            if sel_type == "Point":
                light.SetPositional(True)
                light.SetConeAngle(180.0)
                light.SetExponent(1.0)
            elif sel_type == "Directional":
                light.SetPositional(False)
            elif sel_type == "Spot":
                light.SetPositional(True)
            self.light_registry[lname]['type'] = sel_type
    
        # Update position (for Point/Spot lights) or gizmo position (for Directional)
        px = lc["PosX"].value()
        py = lc["PosY"].value()
        pz = lc["PosZ"].value()
        
        # Always update gizmo position
        self.light_registry[lname]['gizmo'].SetPosition(px, py, pz)
        
        # For directional lights, position is just a reference point for the gizmo
        # The actual light direction is what matters
        light.SetPosition(px, py, pz)
    
        # Update direction (focal point)
        dx = lc["DirX"].value()
        dy = lc["DirY"].value()
        dz = lc["DirZ"].value()
        import math
        mag = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
        nd = (dx/mag, dy/mag, dz/mag)
        fp = (px + nd[0], py + nd[1], pz + nd[2])
        light.SetFocalPoint(*fp)
    
        # Update intensity
        light.SetIntensity(lc["Intensity"].value())
    
        # RGB color
        r = lc["Red"].value() / 100.0
        g = lc["Green"].value() / 100.0
        b = lc["Blue"].value() / 100.0
        light.SetDiffuseColor(r, g, b)
    
        # Update slider labels
        self.lighting_slider_value_labels["Red"].setText(f"{lc['Red'].value()}%")
        self.lighting_slider_value_labels["Green"].setText(f"{lc['Green'].value()}%")
        self.lighting_slider_value_labels["Blue"].setText(f"{lc['Blue'].value()}%")
    
        # Update spot light parameters
        light.SetConeAngle(lc["SpotAngle"].value())
        light.SetExponent(lc["Penumbra"].value())
    
        self.vtk_app.render_all()

    # ---- math helpers ----
    def _world_pos_from_actor(self, actor):
            m = vtk.vtkMatrix4x4()
            actor.GetMatrix(m)  # includes UserTransform, Position, Orientation, Scale
            return (m.GetElement(0, 3), m.GetElement(1, 3), m.GetElement(2, 3))


    def _dir_from_transform(self, transform: vtk.vtkTransform):
        """Direction vector corresponding to -Z axis of the transform."""
        import numpy as np
        m = transform.GetMatrix()
        # Third column is transformed Z axis; -Z is 'forward'
        z = np.array([m.GetElement(0,2), m.GetElement(1,2), m.GetElement(2,2)], dtype=float)
        fwd = -z
        n = np.linalg.norm(fwd) or 1.0
        return (float(fwd[0]/n), float(fwd[1]/n), float(fwd[2]/n))

    def _set_actor_orientation_from_dir(self, actor: vtk.vtkActor, direction):
        """Orient an actor so its -Z points along 'direction'."""
        import math
        dx, dy, dz = direction
        # Yaw (Z-up world): atan2(y, x)
        yaw = math.degrees(math.atan2(dy, dx))
        # Pitch: angle from horizontal plane
        horiz = math.sqrt(dx*dx + dy*dy) or 1.0
        pitch = -math.degrees(math.atan2(dz, horiz))
        actor.SetOrientation(pitch, 0.0, yaw)

    def _light_direction_from_light(self, light):
        """Compute normalized direction from light position to its focal point."""
        import math
        px, py, pz = light.GetPosition()
        fx, fy, fz = light.GetFocalPoint()
        dx, dy, dz = (fx - px, fy - py, fz - pz)
        mag = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
        return (dx/mag, dy/mag, dz/mag)

    def setup_transform_widget(self, actor):
        """Sets up the transform widget for the selected actor."""
        self.current_selected_actor = actor

        self._remove_scale_key_observers()
        self._clear_uniform_scale_hints()

        if self.transform_widget:
            self.transform_widget.Off()
            self.transform_widget = None

        if actor and self.toggle_gizmo_action.isChecked():
            light, _ = self.get_selected_light()
            is_light = light is not None

            mapper = actor.GetMapper()
            if not mapper or not mapper.GetInput():
                return
            original_bounds = mapper.GetInput().GetBounds()
            world_bounds = actor.GetBounds()

            # Lights: ensure no lingering user transform that could offset gizmo
            if is_light:
                actor.SetUserTransform(None)
            else:
                actor_transform = actor.GetUserTransform()
                if not actor_transform:
                    actor_transform = vtk.vtkTransform()
                    actor.SetUserTransform(actor_transform)

            if self.current_transform_mode == 'translate':
                self.setup_translate_handle_widget(actor)
            elif self.current_transform_mode == 'rotate':
                self.setup_rotate_widget(actor, world_bounds if is_light else original_bounds)
            elif self.current_transform_mode == 'scale':
                self.setup_scale_widget(actor, world_bounds if is_light else original_bounds)
            elif self.current_transform_mode == 'transform':
                self.setup_all_transform_widget(actor, world_bounds if is_light else original_bounds)

            if self.transform_widget and self.current_transform_mode != 'translate' and not is_light:
                rep = self.transform_widget.GetRepresentation()
                rep.SetTransform(actor.GetUserTransform())

            self.vtk_app.render_all()

    def setup_translate_handle_widget(self, actor):
        """Sets up a vtkImplicitPlaneWidget2 for translation, showing a circle."""
        self.transform_widget = vtk.vtkImplicitPlaneWidget2()
        self.transform_widget.SetInteractor(self.vtk_app.interactor)

        rep = vtk.vtkImplicitPlaneRepresentation()
        rep.SetPlaceFactor(1.25)
        rep.PlaceWidget(actor.GetBounds())

        # IMPORTANT: use true world position from the actor's matrix
        world_pos = self._world_pos_from_actor(actor)
        rep.SetOrigin(world_pos)

        camera = self.vtk_app.renderer.GetActiveCamera()
        rep.SetNormal(camera.GetDirectionOfProjection())

        rep.DrawPlaneOff()
        rep.SetOutlineTranslation(True)
        rep.SetScaleEnabled(False)
        rep.SetTubing(True)
        rep.GetOutlineProperty().SetColor(1.0, 0.8, 0.0)
        rep.GetOutlineProperty().SetLineWidth(4)
        rep.GetSelectedOutlineProperty().SetColor(1.0, 1.0, 0.0)
        rep.GetEdgesProperty().SetOpacity(0)

        self.transform_widget.SetRepresentation(rep)
        self.transform_widget.AddObserver("InteractionEvent", self.on_translate_plane_interact)
        self.transform_widget.AddObserver("StartInteractionEvent", self.on_translate_start)
        self.transform_widget.AddObserver("EndInteractionEvent", self.on_translate_end)
        self.transform_widget.On()

    def on_translate_start(self, widget, event):
        """Ensure the plane faces the camera at grab time and recenter on true world pos."""
        rep = widget.GetRepresentation()
        camera = self.vtk_app.renderer.GetActiveCamera()
        rep.SetNormal(camera.GetDirectionOfProjection())

        actor = self.get_selected_actor()
        if actor:
            rep.SetOrigin(self._world_pos_from_actor(actor))
            # NEW: record transform before interaction
            self.record_transform_begin(actor)

    def on_translate_end(self, widget, event):
        """After drag ends, update widget bounds and origin to the actor's new world position."""
        actor = self.get_selected_actor()
        if not actor:
            return
        rep = widget.GetRepresentation()
        rep.PlaceWidget(actor.GetBounds())
        rep.SetOrigin(self._world_pos_from_actor(actor))
        # NEW: record transform after interaction
        self.record_transform_end(actor)
        self.vtk_app.render_all()

    def on_translate_plane_interact(self, widget, event):
        """Callback for the implicit plane widget interaction (move)."""
        actor = self.get_selected_actor()
        if not actor:
            return

        rep = widget.GetRepresentation()
        new_origin = rep.GetOrigin()

        light, lname = self.get_selected_light()
        if light:
            light_type = self.light_registry[lname]['type']

            if light_type == "Directional":
                # CHANGE: keep focal point fixed; moving position changes direction.
                fx, fy, fz = light.GetFocalPoint()
                actor.SetPosition(*new_origin)
                light.SetPosition(*new_origin)
                light.SetFocalPoint(fx, fy, fz)
            else:
                # Point/Spot lights: keep direction vector while moving
                px0, py0, pz0 = light.GetPosition()
                fx, fy, fz = light.GetFocalPoint()
                dir_vec = (fx - px0, fy - py0, fz - pz0)

                actor.SetPosition(*new_origin)
                light.SetPosition(*new_origin)
                light.SetFocalPoint(
                    new_origin[0] + dir_vec[0],
                    new_origin[1] + dir_vec[1],
                    new_origin[2] + dir_vec[2]
                )

            self._populate_light_controls(light, self.light_registry[lname]['type'])
            self.vtk_app.render_all()
            return
    
        # Mesh branch (unchanged)...
        current_tf = actor.GetUserTransform()
        if not current_tf:
            current_tf = vtk.vtkTransform()
            current_tf.Identity()
    
        m = vtk.vtkMatrix4x4()
        m.DeepCopy(current_tf.GetMatrix())
        m.SetElement(0, 3, new_origin[0])
        m.SetElement(1, 3, new_origin[1])
        m.SetElement(2, 3, new_origin[2])
    
        new_tf = vtk.vtkTransform()
        new_tf.SetMatrix(m)
        actor.SetUserTransform(new_tf)
    
        self.update_properties_panel(actor)
        self.vtk_app.render_all()

    def setup_rotate_widget(self, actor, bounds):
        """Sets up rotation-only widget."""
        self.transform_widget = vtk.vtkBoxWidget2()
        self.transform_widget.SetInteractor(self.vtk_app.interactor)
        
        rep = vtk.vtkBoxRepresentation()
        rep.SetPlaceFactor(1.25)
        
        # Place widget on the untransformed bounds first.
        # The calling function (setup_transform_widget) will apply the actor's transform to the representation.
        rep.PlaceWidget(bounds)
        
        # Style for rotation: make faces and handles invisible, show only the outline.
        rep.HandlesOn()
        rep.GetOutlineProperty().SetColor(0.3, 1.0, 0.3) # Green outline
        rep.GetOutlineProperty().SetLineWidth(2)
        rep.GetSelectedOutlineProperty().SetColor(1.0, 1.0, 0.0) # Yellow when selected
        rep.GetFaceProperty().SetOpacity(0.0) # Invisible faces
        rep.GetHandleProperty().SetOpacity(0.0) # Invisible handles
        
        self.transform_widget.SetRepresentation(rep)
        
        # Enable ONLY rotation
        self.transform_widget.SetTranslationEnabled(0)
        self.transform_widget.SetRotationEnabled(1)
        self.transform_widget.SetScalingEnabled(0)
        
        self.transform_widget.AddObserver("InteractionEvent", self.on_transform_widget_interact)
        self.transform_widget.AddObserver("StartInteractionEvent", self.on_transform_start)
        self.transform_widget.AddObserver("EndInteractionEvent", self.on_transform_end)
        self.transform_widget.On()

    def on_transform_start(self, widget, event):
        actor = self.get_selected_actor()
        light, _ = self.get_selected_light()
        if actor and not light:
            self.record_transform_begin(actor)

    def on_transform_end(self, widget, event):
        actor = self.get_selected_actor()
        light, _ = self.get_selected_light()
        if actor and not light:
            self.record_transform_end(actor)

    def setup_scale_widget(self, actor, bounds):
        """Sets up scaling-only widget."""
        self.transform_widget = vtk.vtkBoxWidget2()
        self.transform_widget.SetInteractor(self.vtk_app.interactor)

        rep = vtk.vtkBoxRepresentation()
        rep.SetPlaceFactor(1.25)
        rep.PlaceWidget(bounds)

        # Style for scaling
        rep.HandlesOn()
        rep.GetHandleProperty().SetColor(1.0, 0.5, 0.2)
        rep.GetSelectedHandleProperty().SetColor(1.0, 1.0, 0.0)
        rep.GetFaceProperty().SetOpacity(0.1)

        self.transform_widget.SetRepresentation(rep)

        # Enable ONLY scaling
        self.transform_widget.SetTranslationEnabled(0)
        self.transform_widget.SetRotationEnabled(0)
        self.transform_widget.SetScalingEnabled(1)

        self.transform_widget.AddObserver("InteractionEvent", self.on_transform_widget_interact)
        self.transform_widget.AddObserver("StartInteractionEvent", self.on_transform_start)
        self.transform_widget.AddObserver("EndInteractionEvent", self.on_transform_end)
        self.transform_widget.On()

        # NEW: Shift = uniform scaling + visual hints
        self._install_scale_key_observers()
        self._ensure_uniform_scale_hints()
        self._update_uniform_scale_hints(actor)

    def setup_all_transform_widget(self, actor, bounds):
        """Sets up widget with all transformations enabled."""
        self.transform_widget = vtk.vtkBoxWidget2()
        self.transform_widget.SetInteractor(self.vtk_app.interactor)
        
        rep = vtk.vtkBoxRepresentation()
        rep.SetPlaceFactor(1.25)
        
        # Place widget FIRST
        rep.PlaceWidget(bounds)
        
        # Default styling
        rep.HandlesOn()
        rep.GetFaceProperty().SetColor(0.5, 0.5, 0.5)
        rep.GetFaceProperty().SetOpacity(0.2)
        rep.GetOutlineProperty().SetColor(0.8, 0.8, 0.8)
        
        self.transform_widget.SetRepresentation(rep)
        
        # Enable all transformations
        self.transform_widget.SetTranslationEnabled(1)
        self.transform_widget.SetRotationEnabled(1)
        self.transform_widget.SetScalingEnabled(1)
        
        self.transform_widget.AddObserver("StartInteractionEvent", self.on_transform_start)
        self.transform_widget.AddObserver("EndInteractionEvent", self.on_transform_end)
        self.transform_widget.On()

    def on_transform_widget_interact(self, widget, event):
        """Callback for the transform widget interaction."""
        actor = self.get_selected_actor()
        if not actor:
            return

        rep = self.transform_widget.GetRepresentation()
        new_transform = vtk.vtkTransform()
        rep.GetTransform(new_transform)

        light, lname = self.get_selected_light()
        if light:
            # Translate vs rotate/scale behavior
            if self.current_transform_mode in ('rotate', 'scale'):
                pos = actor.GetPosition()
            else:
                pos = new_transform.GetPosition()

            actor.SetPosition(*pos)

            lt = self.light_registry[lname]['type'].lower()
            if lt == "directional" and self.current_transform_mode == 'translate':
                # Keep focal fixed; only move position to change direction
                fx, fy, fz = light.GetFocalPoint()
                light.SetPosition(*pos)
                light.SetFocalPoint(fx, fy, fz)
            else:
                # Update position and, if not point, aim from transform
                light.SetPosition(*pos)
                if lt != "point":
                    dir_vec = self._dir_from_transform(new_transform)
                    fp = (pos[0] + dir_vec[0], pos[1] + dir_vec[1], pos[2] + dir_vec[2])
                    light.SetFocalPoint(*fp)

            # Re-place widget so the box remains centered on the gizmo
            rep.PlaceWidget(actor.GetBounds())

            self._populate_light_controls(light, self.light_registry[lname]['type'])
            self.vtk_app.render_all()
            return

        # Mesh branch
        if self.current_transform_mode == 'scale' and self._scale_uniform_lock:
            # Force uniform scale when Shift is held
            uni_tf = self._make_uniform_scale_transform(new_transform)
            actor.SetUserTransform(uni_tf)
            try:
                rep.SetTransform(uni_tf)  # keep widget in sync
            except Exception:
                pass
            self._update_uniform_scale_hints(actor)
        else:
            actor.SetUserTransform(new_transform)
            # Update hints (will hide if not in scale/Shift)
            self._update_uniform_scale_hints(actor)

        self.update_properties_panel(actor)
        self.vtk_app.render_all()
        

    def create_transform_mode_buttons(self):
        self.transform_button_container = QtWidgets.QWidget(self.vtkWidget)
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self.transform_button_group = QtWidgets.QButtonGroup(self)
        self.transform_button_group.setExclusive(True)

        # UPDATED: use ICON_DIR paths
        modes = [
            ('translate', 'Move (G)', 'translate_icon.png', QtWidgets.QStyle.SP_ArrowUp),
            ('rotate',    'Rotate (R)', 'rotate_icon.png', QtWidgets.QStyle.SP_BrowserReload),
            ('scale',     'Scale (S)',  'scale_icon.png', QtWidgets.QStyle.SP_DesktopIcon),
            ('transform', 'All',        'all_icon.png', QtWidgets.QStyle.SP_DirOpenIcon)
        ]

        for mode, name, filename, fallback_enum in modes:
            path = os.path.join(ICON_DIR, filename)
            button = QtWidgets.QToolButton()
            button.setIcon(_icon(path, self.style(), fallback_enum))
            button.setIconSize(QtCore.QSize(28, 28))
            button.setFixedSize(40, 40)
            button.setCheckable(True)
            button.setToolTip(name)
            button.setAutoRaise(True)
            layout.addWidget(button)
            self.transform_button_group.addButton(button)
            button.clicked.connect(lambda checked, m=mode: self.set_transform_mode(m))

        buttons = self.transform_button_group.buttons()
        if buttons:
            buttons[0].setChecked(True)

        self.transform_button_container.setLayout(layout)
        self.transform_button_container.move(10, 10)
        self.transform_button_container.show()

    def set_transform_mode(self, mode):
        """Sets the current transform mode and updates the gizmo."""
        self.current_transform_mode = mode
        print(f"Transform mode set to: {mode}")
        self.statusBar().showMessage(f"Transform mode: {mode.capitalize()}")
        
        # If an add tool is active, ignore gizmo rebuild
        if self.current_tool:
            return
        # Recreate the widget with the new mode if an actor is selected
        if self.current_selected_actor:
            self.setup_transform_widget(self.current_selected_actor)

    def add_actor_with_name(self, name: str, actor: vtk.vtkActor):
        if not actor:
            return
        # Generate unique name
        count = len([key for key in self.object_registry if key.startswith(name)]) + 1
        unique_name = f"{name}_{count}" if count > 1 else name

        # Register and add to scene
        self.object_registry[unique_name] = actor
        actor.SetUserTransform(None)
        self.vtk_app.add_actor_to_scene(actor)

        # Tree-aware outliner entry
        if isinstance(self.scene_outliner, QtWidgets.QTreeWidget):
            # Place under currently selected collection, or default "Collection"
            sel = self.scene_outliner.selectedItems()
            if sel and sel[0].data(0, QtCore.Qt.UserRole) == "collection":
                parent = sel[0]
            else:
                parent = self.ensure_collection("Collection")

            item = self._make_object_item(unique_name, "mesh", parent)
            self.scene_outliner.setCurrentItem(item)
            self.on_outliner_selection_changed(item)
        else:
            # Fallback for legacy QListWidget UIs
            list_item = QtWidgets.QListWidgetItem(unique_name)
            self.scene_outliner.addItem(list_item)
            self.scene_outliner.setCurrentItem(list_item)
            self.on_outliner_selection_changed(list_item)

        self.update_scene_totals()

    # NEW: start the Add Cube interactive tool
    def activate_add_cube_tool(self):
        if hasattr(self, "vertex_edit_action"):
            self.vertex_edit_action.setChecked(False)
        # If running, stop then restart
        if self.current_tool:
            self.current_tool.stop(cancel=True)
            self.current_tool = None
        self.current_tool = AddCubeTool(self)
        self.current_tool.start()

    def resizeEvent(self, event):
        """Handle window resize to reposition overlay widgets."""
        super().resizeEvent(event)
        if hasattr(self, 'transform_button_container'):
            # Position the transform buttons in top-left corner (adjust as needed)
            self.transform_button_container.move(10, 10)
        self._position_camera_label()

    def open_camera_dialog(self):
        dialog = CameraPropertiesDialog(self.vtk_app.renderer, self)
        dialog.exec_()
        
    def toggle_gizmo(self, checked):
        if self.transform_widget:
            self.setup_transform_widget(self.get_selected_actor())

class AddCubeTool:
    """
    Blender-like interactive Add Cube tool.
    Stages:
      - Base (width/depth on XY plane @ Z=0)
      - Height (along Z, can be negative)
    Modifiers: Shift(square), Ctrl(snap), Alt(from center)
    Input: W=..., D=..., H=..., plain numbers, Tab cycles, Enter confirms, Esc/Right-click cancel.
    """
    def __init__(self, main: MainWindow):
        self.main = main
        self.vtk_app = main.vtk_app
        self.renderer = self.vtk_app.renderer
        self.window = self.vtk_app.window
        self.iren = self.vtk_app.interactor

        # State
        self.stage = "await_first"  # await_first -> base -> height
        self.first_pt = None  # world point on Z=0 plane
        self.base_center = (0.0, 0.0, 0.0)
        self.width = 0.0
        self.depth = 0.0
        self.height = 0.0

        # Modifiers
        self.shift = False
        self.ctrl = False
        self.alt = False

        # Numeric input
        self.numeric_mode = False
        self.active_field = "W"  # W/D in base, H in height
        self.num_buffer = ""

        # Ghost actors
        self.ghost_cube_source = None
        self.ghost_cube_actor = None

        # Overlay label
        self.dim_label = None

        # Observers
        self.obs = []
        self.old_style = None
        self.axes_was_enabled = False

    # ----- lifecycle -----
    def start(self):
        # Cursor to crosshair
        self.main.vtkWidget.setCursor(QtCore.Qt.CrossCursor)

        # Disable camera style (so our tool consumes events)
        try:
            self.old_style = self.iren.GetInteractorStyle()
            style = vtk.vtkInteractorStyleUser()
            self.iren.SetInteractorStyle(style)
        except Exception:
            self.old_style = None

        # Temporarily disable orientation axes widget to avoid conflicting clicks
        try:
            if self.vtk_app.axes_widget:
                self.axes_was_enabled = bool(self.vtk_app.axes_widget.GetEnabled())
                self.vtk_app.axes_widget.SetEnabled(0)
        except Exception:
            pass

        # Disable any transform gizmo while constructing
        try:
            if self.main.transform_widget:
                self.main.transform_widget.Off()
                self.main.transform_widget = None
        except Exception:
            pass

        # Overlay for dimensions
        self.dim_label = QtWidgets.QLabel(self.main.vtkWidget)
        self.dim_label.setStyleSheet("QLabel{background: rgba(20,20,20,150); color: white; border: 1px solid #555; padding: 4px; font: 10pt 'Consolas';}")
        self.dim_label.hide()

        # Hook events
        self._add_obs('MouseMoveEvent', self.on_mouse_move)
        self._add_obs('LeftButtonPressEvent', self.on_left_click)
        self._add_obs('RightButtonPressEvent', self.on_right_click)
        self._add_obs('KeyPressEvent', self.on_key_press)
        self._add_obs('KeyReleaseEvent', self.on_key_release)
        self._add_obs('CharEvent', self.on_char)

        self._update_overlay_text()
        self.vtk_app.render_all()
        self.main.statusBar().showMessage("Add Cube: click to set first corner (or center with Alt). Shift: square, Ctrl: snap, Esc: cancel")

    def stop(self, cancel=False):
        # Remove ghost
        try:
            if self.ghost_cube_actor:
                self.renderer.RemoveActor(self.ghost_cube_actor)
                if self.ghost_cube_actor in self.vtk_app.actors:
                    self.vtk_app.actors.remove(self.ghost_cube_actor)
        except Exception:
            pass

        # Remove observers
        for (ev, tag) in self.obs:
            try:
                self.iren.RemoveObserver(tag)
            except Exception:
                pass
        self.obs.clear()

        # Restore interactor style
        try:
            if self.old_style:
                self.iren.SetInteractorStyle(self.old_style)
        except Exception:
            pass

        # Restore axes widget
        try:
            if self.vtk_app.axes_widget:
                self.vtk_app.axes_widget.SetEnabled(1 if self.axes_was_enabled else 0)
        except Exception:
            pass

        # Restore cursor
        try:
            self.main.vtkWidget.unsetCursor()
        except Exception:
            pass

        # Hide overlay
        if self.dim_label:
            self.dim_label.hide()
            self.dim_label.deleteLater()
            self.dim_label = None

        # Clear status
        if cancel:
            self.main.statusBar().showMessage("Add Cube: cancelled")
        else:
            self.main.statusBar().showMessage("Add Cube: done")

        # Ensure a render
        self.vtk_app.render_all()

    def _add_obs(self, event_name, cb):
        tag = self.iren.AddObserver(event_name, cb, 1.0)
        self.obs.append((event_name, tag))

    # ----- geometry helpers -----
    def _display_to_world(self, x, y, z_ndc):
        # z_ndc in [0..1] display depth
        self.renderer.SetDisplayPoint(float(x), float(y), float(z_ndc))
        self.renderer.DisplayToWorld()
        wp = self.renderer.GetWorldPoint()
        if wp[3] == 0.0:
            return (0.0, 0.0, 0.0)
        return (wp[0] / wp[3], wp[1] / wp[3], wp[2] / wp[3])

    def _screen_to_plane_z0(self, sx, sy):
        # Ray from near->far, intersect with plane z=0
        p0 = self._display_to_world(sx, sy, 0.0)
        p1 = self._display_to_world(sx, sy, 1.0)
        # Line p0 + t*(p1-p0), find z=0
        dx, dy, dz = (p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2])
        if abs(dz) < 1e-9:
            t = 0.0
        else:
            t = -p0[2]/dz
        x = p0[0] + dx*t
        y = p0[1] + dy*t
        return (x, y, 0.0)

    def _apply_snapping(self, value):
        inc = float(self.main.snap_increment or 1.0)
        return round(value / inc) * inc

    def _build_ghost_if_needed(self):
        if self.ghost_cube_source:
            return
        self.ghost_cube_source = vtk.vtkCubeSource()
        self.ghost_cube_source.SetXLength(0.0)
        self.ghost_cube_source.SetYLength(0.0)
        self.ghost_cube_source.SetZLength(1e-3)
        self.ghost_cube_source.SetCenter(0.0, 0.0, 0.0)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.ghost_cube_source.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(0.8, 0.8, 1.0)
        prop.SetOpacity(0.25)
        prop.SetRepresentationToSurface()
        prop.EdgeVisibilityOn()
        prop.SetEdgeColor(0.6, 0.8, 1.0)
        prop.SetSpecular(0.0)
        prop.SetDiffuse(1.0)
        actor.PickableOff()
        self.ghost_cube_actor = actor
        self.renderer.AddActor(actor)

    def _update_overlay_text(self, evt_pos=None):
        if not self.dim_label:
            return
        W = abs(self.width)
        D = abs(self.depth)
        H = abs(self.height) if self.stage == "height" else 0.0
        txt = f"W {W:.3f}  ×  D {D:.3f}  ×  H {H:.3f}"
        if self.numeric_mode and self.num_buffer:
            txt += f"   [{self.active_field}={self.num_buffer}]"
        self.dim_label.setText(txt)
        # Position near cursor
        if evt_pos:
            mx, my = evt_pos
        else:
            mx, my = self.iren.GetEventPosition()
        # Qt origin at top-left; VTK gives bottom-left, so flip Y
        h = self.main.vtkWidget.height()
        self.dim_label.adjustSize()
        w_lbl = self.dim_label.width()
        h_lbl = self.dim_label.height()
        x = int(mx + 16)
        y = int(h - my - 16 - h_lbl)
        self.dim_label.move(max(0, min(self.main.vtkWidget.width()-w_lbl, x)),
                           max(0, min(self.main.vtkWidget.height()-h_lbl, y)))
        self.dim_label.show()

    def _compute_base_from_points(self, anchor, cursor):
        ax, ay, _ = anchor
        cx, cy, _ = cursor

        if self.alt:
            # from center
            hx = cx - ax
            hy = cy - ay
            if self.shift:
                mag = max(abs(hx), abs(hy))
                # preserve quadrant sign
                sx = 1.0 if hx >= 0 else -1.0
                sy = 1.0 if hy >= 0 else -1.0
                hx = sx * mag
                hy = sy * mag
            if self.ctrl:
                hx = self._apply_snapping(hx)
                hy = self._apply_snapping(hy)
            center = (ax, ay, 0.0)
            width = 2.0 * hx
            depth = 2.0 * hy
        else:
            # from corner
            dx = cx - ax
            dy = cy - ay
            if self.shift:
                mag = max(abs(dx), abs(dy))
                sx = 1.0 if dx >= 0 else -1.0
                sy = 1.0 if dy >= 0 else -1.0
                dx = sx * mag
                dy = sy * mag
            if self.ctrl:
                dx = self._apply_snapping(dx)
                dy = self._apply_snapping(dy)
            center = (ax + dx*0.5, ay + dy*0.5, 0.0)
            width = dx
            depth = dy

        return center, width, depth

    def _update_ghost_base(self):
        if not self.ghost_cube_source:
            return
        # Flat rectangle (height ~ 0)
        W = abs(self.width)
        D = abs(self.depth)
        self.ghost_cube_source.SetXLength(max(W, 0.0))
        self.ghost_cube_source.SetYLength(max(D, 0.0))
        self.ghost_cube_source.SetZLength(1e-3)
        self.ghost_cube_source.SetCenter(self.base_center[0], self.base_center[1], 0.0)
        self.ghost_cube_source.Update()
        self.vtk_app.render_all()

    def _update_ghost_height(self):
        if not self.ghost_cube_source:
            return
        # Height along Z (can be negative). Center z = H/2
        W = abs(self.width)
        D = abs(self.depth)
        H = abs(self.height)
        cz = 0.5 * self.height  # signed half height
        self.ghost_cube_source.SetXLength(max(W, 0.0))
        self.ghost_cube_source.SetYLength(max(D, 0.0))
        self.ghost_cube_source.SetZLength(max(H, 1e-6))
        self.ghost_cube_source.SetCenter(self.base_center[0], self.base_center[1], cz)
        self.ghost_cube_source.Update()
        self.vtk_app.render_all()

    # ----- finalize -----
    def _commit_cube(self):
        # Build a real cube with origin (object pivot) at base center on plane (z=0)
        W = float(abs(self.width))
        D = float(abs(self.depth))
        H = float(abs(self.height))

        # Avoid degenerate dimensions
        if W < 1e-9 or D < 1e-9:
            return False

        src = vtk.vtkCubeSource()
        src.SetXLength(W)
        src.SetYLength(D)
        # Allow zero height to create a plane-like mesh, otherwise use |H|
        src.SetZLength(max(H, 1e-9))
        # Center shifted so that actor's origin at z=0 is the base center:
        signed_half = 0.5 * self.height
        src.SetCenter(0.0, 0.0, signed_half)
        src.Update()

        mapper = self.main.vtk_app.create_mapper(src)
        actor = self.main.vtk_app.create_actor(mapper)
        # Put actor origin at base center on plane
        actor.SetPosition(self.base_center[0], self.base_center[1], 0.0)

        # Undoable add
        self.main.undo_stack.push(AddActorCommand(self.main, "cube", actor))
        return True

    # ----- events -----
    def _in_axes_widget_region(self, x, y):
        try:
            if not self.vtk_app.axes_widget:
                return False
            vp = self.vtk_app.axes_widget.GetViewport()
            w, h = self.window.GetSize()
            x_min = int(vp[0]*w); y_min = int(vp[1]*h)
            x_max = int(vp[2]*w); y_max = int(vp[3]*h)
            return x_min <= x <= x_max and y_min <= y <= y_max
        except Exception:
            return False

    def on_mouse_move(self, obj, evt):
        x, y = self.iren.GetEventPosition()
        if self._in_axes_widget_region(x, y):
            return

        world = self._screen_to_plane_z0(x, y)

        if self.stage == "base" and self.first_pt is not None and not self.numeric_mode:
            self.base_center, self.width, self.depth = self._compute_base_from_points(self.first_pt, world)
            self._build_ghost_if_needed()
            self._update_ghost_base()
            self._update_overlay_text((x, y))
        elif self.stage == "height" and not self.numeric_mode:
            # Height from vertical mouse delta (screen), up -> positive Z
            self.renderer.SetWorldPoint(self.base_center[0], self.base_center[1], 0.0, 1.0)
            self.renderer.WorldToDisplay()
            bx, by, _ = self.renderer.GetDisplayPoint()
            dy_px = (by - y)  # invert so dragging up increases height
            scale = self._vertical_world_units_per_pixel()  # camera-aware scale
            h = dy_px * scale
            if self.ctrl:
                h = self._apply_snapping(h)
            self.height = h
            self._update_ghost_height()
            self._update_overlay_text((x, y))

    def _world_units_per_pixel_at_screen_point(self, sx, sy):
        # Sample two nearby display points and measure world distance
        pA = self._display_to_world(sx, sy, 0.5)
        pB = self._display_to_world(sx, sy+1.0, 0.5)
        # distance in world between 1px move
        d = ((pA[0]-pB[0])**2 + (pA[1]-pB[1])**2 + (pA[2]-pB[2])**2) ** 0.5
        # Use vertical pixel heuristics
        return max(d, 1e-3)

    def on_left_click(self, obj, evt):
        x, y = self.iren.GetEventPosition()
        if self._in_axes_widget_region(x, y):
            return

        if self.stage == "await_first":
            self.first_pt = self._screen_to_plane_z0(x, y)
            self.stage = "base"
            # If Alt(center), first_pt acts as center; else as corner
            # Initialize preview with zero size
            self.base_center = (self.first_pt[0], self.first_pt[1], 0.0)
            self.width = 0.0
            self.depth = 0.0
            self._build_ghost_if_needed()
            self._update_ghost_base()
            self._update_overlay_text((x, y))
            self.main.statusBar().showMessage("Add Cube: move to size base, click/Enter to confirm. Shift=square, Ctrl=snap, Alt=center")
        elif self.stage == "base":
            # Confirm base rectangle
            if abs(self.width) < 1e-9 and abs(self.depth) < 1e-9:
                return
            self.stage = "height"
            self.height = 0.0
            # Keep ghost, switch to height preview
            self._update_ghost_height()
            self._update_overlay_text((x, y))
            self.main.statusBar().showMessage("Add Cube: set height (drag up/down). Click/Enter to confirm. Esc to go back.")
        elif self.stage == "height":
            # Confirm height and create cube
            ok = self._commit_cube()
            self.stop(cancel=not ok)
            self.main.current_tool = None

    def on_right_click(self, obj, evt):
        # Cancel immediately
        self.stop(cancel=True)
        self.main.current_tool = None

    def on_key_press(self, obj, evt):
        key = self.iren.GetKeySym() or ""
        if key in ("Shift_L", "Shift_R"):
            self.shift = True
        elif key in ("Control_L", "Control_R"):
            self.ctrl = True
        elif key in ("Alt_L", "Alt_R"):
            self.alt = True
        elif key in ("Return", "KP_Enter"):
            # Confirm stage
            if self.stage == "base":
                if abs(self.width) >= 1e-9 or abs(self.depth) >= 1e-9:
                    self.stage = "height"
                    self.height = 0.0
                    self._update_ghost_height()
            elif self.stage == "height":
                ok = self._commit_cube()
                self.stop(cancel=not ok)
                self.main.current_tool = None
        elif key == "Escape":
            if self.stage == "height":
                # First Esc -> back to base
                self.stage = "base"
                self.height = 0.0
                self._update_ghost_base()
                self._update_overlay_text()
                self.main.statusBar().showMessage("Add Cube: back to base. Esc again to cancel.")
            else:
                self.stop(cancel=True)
                self.main.current_tool = None
        elif key == "Tab":
            if self.stage == "base":
                self.active_field = "D" if self.active_field == "W" else "W"
            else:
                self.active_field = "H"
            self.numeric_mode = True
            # keep buffer; show overlay tag
            self._update_overlay_text()
        else:
            # Start numeric mode if alnum
            if key.lower() in ("w", "d", "h"):
                field = key.upper()
                if self.stage == "base" and field in ("W", "D"):
                    self.active_field = field
                    self.numeric_mode = True
                    self.num_buffer = ""
                    self._update_overlay_text()
                elif self.stage == "height" and field == "H":
                    self.active_field = field
                    self.numeric_mode = True
                    self.num_buffer = ""
                    self._update_overlay_text()

    def on_key_release(self, obj, evt):
        key = self.iren.GetKeySym() or ""
        if key in ("Shift_L", "Shift_R"):
            self.shift = False
        elif key in ("Control_L", "Control_R"):
            self.ctrl = False
        elif key in ("Alt_L", "Alt_R"):
            self.alt = False

    def on_char(self, obj, evt):
        # Handle numeric entry
        ch = self.iren.GetKeySym() or ""
        kc = self.iren.GetKeyCode()
        # Filter digits, dot, minus, backspace, equals
        if kc in ('\b', '\x7f'):  # backspace/delete
            if self.num_buffer:
                self.num_buffer = self.num_buffer[:-1]
                self._apply_numeric()
            return
        if kc == '\r' or kc == '\n':
            # handled in key press
            return
        if kc == '\t':
            return
        if kc == '=':
            # ignore '=' char
            return

        if (kc.isdigit() or kc in ('.', '-',)) or (kc.isalpha()):
            # If user types field prefix like W, D, H
            if kc.isalpha():
                up = kc.upper()
                if self.stage == "base" and up in ("W", "D"):
                    self.active_field = up
                    self.numeric_mode = True
                    self.num_buffer = ""
                    self._update_overlay_text()
                    return
                if self.stage == "height" and up == "H":
                    self.active_field = up
                    self.numeric_mode = True
                    self.num_buffer = ""
                    self._update_overlay_text()
                    return
                # ignore other letters (units suffix 'm' ignored)
                return
            # append numeric characters
            if not self.numeric_mode:
                self.numeric_mode = True
                self.num_buffer = ""
            self.num_buffer += kc
            self._apply_numeric()

    def _parse_number(self, s: str) -> float:
        # Strip trailing unit letters (e.g., 'm') and parse
        import re
        m = re.match(r'^\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)', s)
        if not m:
            return 0.0
        try:
            return float(m.group(1))
        except Exception:
            return 0.0
        
    def _vertical_world_units_per_pixel(self):
        """World units per vertical screen pixel at the base center, camera-aware."""
        cam = self.renderer.GetActiveCamera()
        _, vp_y0, _, vp_y1 = self.renderer.GetViewport()
        win_w, win_h = self.window.GetSize()
        # Effective viewport pixel height
        vp_h = max(int((vp_y1 - vp_y0) * win_h), 1)

        import math
        if cam.GetParallelProjection():
            # ParallelScale is half the height of the viewport in world units
            units_per_px = (cam.GetParallelScale() * 2.0) / vp_h
        else:
            # Perspective: 2 * d * tan(fov/2) over viewport height
            bx, by, bz = self.base_center
            cx, cy, cz = cam.GetPosition()
            d = math.sqrt((cx - bx)**2 + (cy - by)**2 + (cz - 0.0)**2)
            fov_rad = math.radians(cam.GetViewAngle())
            units_per_px = (2.0 * d * math.tan(fov_rad * 0.5)) / vp_h

        # Slight boost so it feels responsive (tweak if needed)
        return max(units_per_px * 1.75, 1e-5)

    def _apply_numeric(self):
        val = self._parse_number(self.num_buffer)
        if self.ctrl:
            val = self._apply_snapping(val)
        if self.stage == "base":
            if self.active_field == "W":
                # width magnitude; keep center fixed, set absolute value
                self.width = val if (self.alt or self.first_pt is None) else (val if self.width >= 0 else -val)
            elif self.active_field == "D":
                self.depth = val if (self.alt or self.first_pt is None) else (val if self.depth >= 0 else -val)
            self._build_ghost_if_needed()
            self._update_ghost_base()
        elif self.stage == "height":
            if self.active_field == "H":
                self.height = val
                self._update_ghost_height()
        self._update_overlay_text()
        self.vtk_app.render_all()

class FaceEditTool:
    """
    Minimal Face Edit Mode:
      - Click a face to select (highlights).
      - Drag to move that face along its normal (moves all vertices of that face).
      - Press Delete (or click overlay button) to delete the face.
    Notes:
      * Edits a single selected mesh actor.
      * No topology fixes; shared vertices move with the face.
      * Undo integrates via existing VertexEditCommand.
    """
    def __init__(self, main_window: MainWindow):
        self.main = main_window
        self.vtk_app = main_window.vtk_app
        self.renderer = self.vtk_app.renderer
        self.window = self.vtk_app.window
        self.iren = self.vtk_app.interactor

        self.active_actor = None
        self.edit_poly = None

        self.cell_picker = vtk.vtkCellPicker()
        self.cell_picker.SetTolerance(0.0005)

        self.selected_cell_id = -1
        self.face_point_ids = []       # [pid...]
        self.face_points_start = []    # [(x,y,z)...] at drag start
        self.face_normal = (0.0, 0.0, 1.0)

        self.dragging = False
        self.drag_start_y = 0
        self.poly_before = None

        # Highlight actor (semi-transparent overlay of the selected face)
        self.highlight_actor = None
        self.highlight_mapper = None

        # Overlay delete button
        self.delete_btn = None

        self.obs = []
        self.old_style = None
        self.axes_was_enabled = False
        self.await_second_click = False

    # ---- lifecycle ----
    def start(self):
        actor = self.main.get_selected_actor()
        light, _ = self.main.get_selected_light()
        if not actor or light or not self.main._is_face_edit_allowed(actor):
            return

        self.active_actor = actor

        # Make an editable copy of the mesh
        src = actor.GetMapper().GetInput()
        poly = self.main.as_polydata(src)
        if poly is None:
            return

        self.edit_poly = vtk.vtkPolyData()
        self.edit_poly.DeepCopy(poly)

        self.poly_before = vtk.vtkPolyData()
        self.poly_before.DeepCopy(self.edit_poly)

        # IMPORTANT: map the editable poly directly so picker CellIds match
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.edit_poly)
        mapper.ScalarVisibilityOff()
        self.active_actor.SetMapper(mapper)

        # Build highlight overlay (shares same points)
        self._build_highlight_actor()

        # Disable camera interaction style to capture drags
        try:
            self.old_style = self.iren.GetInteractorStyle()
            self.iren.SetInteractorStyle(vtk.vtkInteractorStyleUser())
        except Exception:
            self.old_style = None

        # Turn off transform gizmo and hide transform mode buttons
        try:
            if self.main.transform_widget:
                self.main.transform_widget.Off()
                self.main.transform_widget = None
            if hasattr(self.main, "transform_button_container") and self.main.transform_button_container:
                self.main.transform_button_container.setVisible(False)
        except Exception:
            pass

        # Disable axes widget to avoid click conflicts
        try:
            if self.vtk_app.axes_widget:
                self.axes_was_enabled = bool(self.vtk_app.axes_widget.GetEnabled())
                self.vtk_app.axes_widget.SetEnabled(0)
        except Exception:
            pass

        self.iren.SetPicker(self.cell_picker)

        # Overlay Delete button
        self.delete_btn = QtWidgets.QPushButton("Delete Face", self.main.vtkWidget)
        self.delete_btn.setStyleSheet(
            "QPushButton{background:rgba(160,25,25,200); color:#fff; border:1px solid #802; padding:4px 8px; font:9pt 'Consolas';}"
            "QPushButton:hover{background:rgba(190,35,35,220);}"
        )
        self.delete_btn.adjustSize()
        self._position_delete_button()
        self.delete_btn.hide()
        self.delete_btn.clicked.connect(self._delete_selected_face)

        # Observers
        self._add_obs("LeftButtonPressEvent", self._on_left_down)
        self._add_obs("LeftButtonReleaseEvent", self._on_left_up)
        self._add_obs("MouseMoveEvent", self._on_mouse_move)
        self._add_obs("KeyPressEvent", self._on_key_press)
        self._add_obs("RenderEvent", self._on_render)

        # First interaction should only select
        self.await_second_click = True
        self.vtk_app.render_all()

    def stop(self, cancel=False):
        # remove highlight
        try:
            if self.highlight_actor:
                self.renderer.RemoveActor(self.highlight_actor)
        except Exception:
            pass
        self.highlight_actor = None
        # remove delete button
        if self.delete_btn:
            self.delete_btn.hide()
            self.delete_btn.deleteLater()
            self.delete_btn = None
        # remove observers
        for (_, tag) in self.obs:
            try:
                self.iren.RemoveObserver(tag)
            except Exception:
                pass
        self.obs.clear()
        # restore style
        try:
            if self.old_style:
                self.iren.SetInteractorStyle(self.old_style)
        except Exception:
            pass
        # restore axes widget
        try:
            if self.vtk_app.axes_widget:
                self.vtk_app.axes_widget.SetEnabled(1 if self.axes_was_enabled else 0)
        except Exception:
            pass

        # push undo snapshot if not cancelled
        if not cancel and self.active_actor and self.edit_poly:
            poly_after = vtk.vtkPolyData()
            poly_after.DeepCopy(self.edit_poly)
            try:
                self.main.undo_stack.push(VertexEditCommand(self.main, self.active_actor, self.poly_before, poly_after))
            except Exception:
                pass

        self.vtk_app.render_all()

    # ---- ui helpers ----
    def _position_delete_button(self):
        if not self.delete_btn:
            return
        m = 10
        w = self.main.vtkWidget.width()
        # place bottom-left
        self.delete_btn.move(m, self.main.vtkWidget.height() - self.delete_btn.height() - m)

    def _on_render(self, *args):
        self._position_delete_button()
        # Ensure highlight keeps matching if actor transform changes
        self._sync_highlight_transform()

    def _add_obs(self, evt, cb):
        tag = self.iren.AddObserver(evt, cb, 1.0)
        self.obs.append((evt, tag))

    # ---- picking / highlight ----
    def _build_highlight_actor(self):
        # The poly for highlight will share points with edit_poly
        self.highlight_points = self.edit_poly.GetPoints()
        self.highlight_poly = vtk.vtkPolyData()
        self.highlight_poly.SetPoints(self.highlight_points)
        self.highlight_cells = vtk.vtkCellArray()
        self.highlight_poly.SetPolys(self.highlight_cells)

        self.highlight_mapper = vtk.vtkPolyDataMapper()
        self.highlight_mapper.SetInputData(self.highlight_poly)
        self.highlight_mapper.ScalarVisibilityOff()
        self.highlight_actor = vtk.vtkActor()
        self.highlight_actor.SetMapper(self.highlight_mapper)

        # Style
        prop = self.highlight_actor.GetProperty()
        prop.SetColor(1.0, 1.0, 0.0)
        prop.SetOpacity(0.35)
        prop.SetLighting(False)
        prop.BackfaceCullingOff()
        prop.FrontfaceCullingOff()
        self.highlight_actor.PickableOff()
        self.highlight_actor.VisibilityOff()

        # IMPORTANT: sync to full actor world transform (Position/Rotation/Scale + UserTransform)
        self._sync_highlight_transform()

        self.renderer.AddActor(self.highlight_actor)

    def _sync_highlight_transform(self):
        if not self.highlight_actor or not self.active_actor:
            return
        try:
            m = vtk.vtkMatrix4x4()
            self.active_actor.GetMatrix(m)  # full world matrix
            t = vtk.vtkTransform()
            t.SetMatrix(m)
            # Drive highlight entirely by this world matrix
            self.highlight_actor.SetUserTransform(t)
            self.highlight_actor.SetPosition(0.0, 0.0, 0.0)
            self.highlight_actor.SetOrientation(0.0, 0.0, 0.0)
            self.highlight_actor.SetScale(1.0, 1.0, 1.0)
        except Exception:
            pass

    def _set_highlight_face(self, cell_id: int):
        self.highlight_cells.Reset()
        self.selected_cell_id = -1
        self.face_point_ids = []
        if cell_id < 0:
            self.highlight_actor.VisibilityOff()
            self.vtk_app.render_all()
            return

        # Robust: fetch points for whatever cell type the dataset has
        if cell_id >= self.edit_poly.GetNumberOfCells():
            self.highlight_actor.VisibilityOff()
            self.vtk_app.render_all()
            return

        id_list = vtk.vtkIdList()
        self.edit_poly.GetCellPoints(cell_id, id_list)
        if id_list.GetNumberOfIds() < 3:
            self.highlight_actor.VisibilityOff()
            self.vtk_app.render_all()
            return

        vtk_ids = vtk.vtkIdList()
        for j in range(id_list.GetNumberOfIds()):
            pid = int(id_list.GetId(j))
            vtk_ids.InsertNextId(pid)
            self.face_point_ids.append(pid)

        self.highlight_cells.InsertNextCell(vtk_ids)
        self.highlight_cells.Modified()
        self.highlight_poly.Modified()
        self.highlight_actor.VisibilityOn()
        self.selected_cell_id = cell_id
        self.vtk_app.render_all()

    # ---- normals / scale ----
    def _compute_face_normal(self):
        if not self.face_point_ids or len(self.face_point_ids) < 3:
            return (0.0, 0.0, 1.0)
        pts = self.edit_poly.GetPoints()
        p0 = pts.GetPoint(self.face_point_ids[0])
        p1 = pts.GetPoint(self.face_point_ids[1])
        p2 = pts.GetPoint(self.face_point_ids[2])
        ux, uy, uz = (p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2])
        vx, vy, vz = (p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2])
        # cross u x v
        nx = uy*vz - uz*vy
        ny = uz*vx - ux*vz
        nz = ux*vy - uy*vx
        import math
        mag = math.sqrt(nx*nx + ny*ny + nz*nz) or 1.0
        return (nx/mag, ny/mag, nz/mag)

    def _units_per_px_at_face(self):
        # Use bbox center of the face
        pts = self.edit_poly.GetPoints()
        cx = cy = cz = 0.0
        n = len(self.face_point_ids) or 1
        for pid in self.face_point_ids:
            x, y, z = pts.GetPoint(pid)
            cx += x; cy += y; cz += z
        cx /= n; cy /= n; cz /= n
        # Same formula as in VertexEditTool
        cam = self.renderer.GetActiveCamera()
        if cam.GetParallelProjection() == 0:
            import math
            px, py, pz = cam.GetPosition()
            dist = math.sqrt((cx-px)**2 + (cy-py)**2 + (cz-pz)**2)
            fov = math.radians(cam.GetViewAngle())
            vp = self.renderer.GetViewport()
            _, win_h = self.window.GetSize()
            pix_h = max(int((vp[3]-vp[1]) * win_h), 1)
            return (2.0 * dist * math.tan(fov*0.5)) / pix_h
        else:
            vp = self.renderer.GetViewport()
            _, win_h = self.window.GetSize()
            pix_h = max(int((vp[3]-vp[1]) * win_h), 1)
            return (self.renderer.GetActiveCamera().GetParallelScale()*2.0) / pix_h

    # ---- events ----
    def _on_left_down(self, obj, evt):
        x, y = self.iren.GetEventPosition()
        self.cell_picker.Pick(x, y, 0, self.renderer)
        picked_actor = self.cell_picker.GetActor()
        cid = int(self.cell_picker.GetCellId()) if self.cell_picker.GetCellId() >= 0 else -1

        # Clicked outside or on a different actor: clear
        if picked_actor is not self.active_actor or cid < 0:
            self._set_highlight_face(-1)
            self.dragging = False
            self.await_second_click = True
            return

        # New face picked -> just highlight, wait for second click to move
        if cid != self.selected_cell_id:
            self._set_highlight_face(cid)
            self.dragging = False
            self.await_second_click = True
            # show delete button when a face is selected
            if self.delete_btn:
                self.delete_btn.show()
            return

        # Same face picked again -> start drag if we've already highlighted once
        if self.await_second_click:
            self.await_second_click = False  # consume the "second click" guard

        # start drag
        self.dragging = True
        self.drag_start_y = y
        self.face_normal = self._compute_face_normal()
        # snapshot start positions for the face's vertices
        pts = self.edit_poly.GetPoints()
        self.face_points_start = [pts.GetPoint(pid) for pid in self.face_point_ids]
        if self.delete_btn:
            self.delete_btn.show()


    def _on_left_up(self, obj, evt):
        self.dragging = False
        # keep selection/highlight

    def _on_mouse_move(self, obj, evt):
        if not self.dragging or self.selected_cell_id < 0:
            return
        _, y = self.iren.GetEventPosition()
        dy_px = (self.drag_start_y - y)  # up => positive
        scale = self._units_per_px_at_face()
        d = dy_px * scale
        nx, ny, nz = self.face_normal
        vx, vy, vz = (nx*d, ny*d, nz*d)
        # apply translation to face vertices (relative to snapshot)
        pts = self.edit_poly.GetPoints()
        for i, pid in enumerate(self.face_point_ids):
            x0, y0, z0 = self.face_points_start[i]
            pts.SetPoint(pid, x0 + vx, y0 + vy, z0 + vz)
        pts.Modified()
        self.edit_poly.Modified()
        self.vtk_app.render_all()

    def _on_key_press(self, obj, evt):
        key = self.iren.GetKeySym() or ""
        if key in ("Delete", "BackSpace", "x", "X"):
            self._delete_selected_face()

    # ---- operations ----
    def _delete_selected_face(self):
        if self.selected_cell_id < 0 or not self.edit_poly:
            return
        # record undo snapshot around delete
        before = vtk.vtkPolyData(); before.DeepCopy(self.edit_poly)

        self.edit_poly.BuildCells()
        self.edit_poly.BuildLinks()
        try:
            self.edit_poly.DeleteCell(self.selected_cell_id)
            self.edit_poly.RemoveDeletedCells()
        except Exception:
            # fallback: rebuild polys skipping the selected id
            new_polys = vtk.vtkCellArray()
            old = self.edit_poly.GetPolys()
            id_list = vtk.vtkIdList()
            idx = 0
            old.InitTraversal()
            while old.GetNextCell(id_list):
                if idx != self.selected_cell_id:
                    new_polys.InsertNextCell(id_list)
                idx += 1
            self.edit_poly.SetPolys(new_polys)
        self.edit_poly.Modified()

        # clear highlight
        self._set_highlight_face(-1)
        if self.delete_btn:
            self.delete_btn.hide()

        self.vtk_app.render_all()

        after = vtk.vtkPolyData(); after.DeepCopy(self.edit_poly)
        try:
            self.main.undo_stack.push(VertexEditCommand(self.main, self.active_actor, before, after))
        except Exception:
            pass


class VertexEditTool:
    """
    Minimal vertex edit tool:
      - works on the currently selected mesh actor
      - click a vertex to select (highlighted as a small sphere)
      - drag mouse to move that vertex in the camera view plane

    Notes:
      * Only single-vertex editing (one at a time).
      * Works for any object type that can be converted to vtkPolyData.
    """
    def __init__(self, main_window):
        self.main = main_window
        self.vtk_app = main_window.vtk_app
        self.renderer = self.vtk_app.renderer
        self.window = self.vtk_app.window
        self.iren = self.vtk_app.interactor

        self.active_actor = None
        self.edit_poly = None

        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.0005)

        self.highlight_actor = None

        self.dragging = False
        self.picked_pid = -1
        self.initial_local_pos = (0.0, 0.0, 0.0)
        self.initial_plane_point_world = (0.0, 0.0, 0.0)
        self.drag_plane_origin = (0.0, 0.0, 0.0)
        self.drag_plane_normal = (0.0, 0.0, 1.0)

        self.actor_matrix = None
        self.actor_matrix_inv = None

        self.obs = []
        self.old_style = None
        self.axes_was_enabled = False

    # ----- lifecycle -----
    def start(self):
        # Work on the currently selected actor
        actor = self.main.get_selected_actor()
        light, _ = self.main.get_selected_light()
        if not actor or light:
            return

        self.active_actor = actor

        # Detach geometry into an editable vtkPolyData copy
        src_data = actor.GetMapper().GetInput()
        poly = self.main.as_polydata(src_data)
        if poly is None:
            return

        self.edit_poly = vtk.vtkPolyData()
        self.edit_poly.DeepCopy(poly)

        self.poly_before = vtk.vtkPolyData()
        self.poly_before.DeepCopy(self.edit_poly)

        tp = vtk.vtkTrivialProducer()
        tp.SetOutput(self.edit_poly)
        new_mapper = self.main.vtk_app.create_mapper(tp)
        self.active_actor.SetMapper(new_mapper)

        # Save actor matrices (world <-> local)
        self.actor_matrix = vtk.vtkMatrix4x4()
        self.active_actor.GetMatrix(self.actor_matrix)
        self.actor_matrix_inv = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(self.actor_matrix, self.actor_matrix_inv)

        # Highlight sphere (hidden until a vertex is selected)
        sphere = vtk.vtkSphereSource()
        sphere.SetThetaResolution(24)
        sphere.SetPhiResolution(24)
        sphere.SetRadius(0.5)
        self.highlight_source = sphere
        s_mapper = vtk.vtkPolyDataMapper()
        s_mapper.SetInputConnection(sphere.GetOutputPort())
        self.highlight_actor = vtk.vtkActor()
        self.highlight_actor.SetMapper(s_mapper)
        # Make highlight follow the same UserTransform as the edited actor
        try:
            self.highlight_actor.SetUserTransform(self.active_actor.GetUserTransform())
        except Exception:
            pass
        prop = self.highlight_actor.GetProperty()
        prop.SetColor(1.0, 1.0, 0.0)
        prop.SetAmbient(0.3)
        prop.SetDiffuse(1.0)
        prop.SetSpecular(0.0)
        self.highlight_actor.PickableOff()
        self.highlight_actor.VisibilityOff()
        self.renderer.AddActor(self.highlight_actor)
        # Adaptive sizing parameters
        self._highlight_pixels = 14
        self._min_world_radius = 0.02
        self._max_world_radius = 5.0
        # Build point locator
        self.point_locator = vtk.vtkPointLocator()
        self.point_locator.SetDataSet(self.edit_poly)
        self.point_locator.BuildLocator()
        # Observe render to rescale sphere every frame
        self._add_obs("RenderEvent", self.on_render)

        # Use a neutral interactor style (no camera rotation) while editing
        try:
            self.old_style = self.iren.GetInteractorStyle()
            self.iren.SetInteractorStyle(vtk.vtkInteractorStyleUser())
        except Exception:
            self.old_style = None

        # Pause transform gizmo and hide transform buttons overlay
        if self.main.transform_widget:
            try:
                self.main.transform_widget.Off()
            except Exception:
                pass
            self.main.transform_widget = None
        try:
            if hasattr(self.main, "transform_button_container") and self.main.transform_button_container:
                self.main.transform_button_container.setVisible(False)
        except Exception:
            pass

        # Disable axes widget to avoid conflicting clicks
        try:
            if self.vtk_app.axes_widget:
                self.axes_was_enabled = bool(self.vtk_app.axes_widget.GetEnabled())
                self.vtk_app.axes_widget.SetEnabled(0)
        except Exception:
            pass

        # Use our picker
        self.iren.SetPicker(self.picker)

        # Hook events
        self._add_obs("LeftButtonPressEvent", self.on_left_down)
        self._add_obs("LeftButtonReleaseEvent", self.on_left_up)
        self._add_obs("MouseMoveEvent", self.on_mouse_move)

        self.vtk_app.render_all()

    def on_render(self, *args):
        # Called each render; keep highlight sphere sized
        self._rescale_highlight()

    def stop(self, cancel=False):
        """End vertex edit session. Push undo command if not cancelled."""
        # Remove highlight actor
        try:
            if self.highlight_actor:
                self.renderer.RemoveActor(self.highlight_actor)
        except Exception:
            pass
        self.highlight_actor = None

        # Detach observers
        for (_, tag) in self.obs:
            try:
                self.iren.RemoveObserver(tag)
            except Exception:
                pass
        self.obs.clear()

        # Restore interactor style
        try:
            if self.old_style:
                self.iren.SetInteractorStyle(self.old_style)
        except Exception:
            pass

        # Restore axes widget
        try:
            if self.vtk_app.axes_widget:
                self.vtk_app.axes_widget.SetEnabled(1 if self.axes_was_enabled else 0)
        except Exception:
            pass

        # Commit undo (only if accepted)
        if not cancel and self.active_actor and self.edit_poly:
            poly_after = vtk.vtkPolyData()
            poly_after.DeepCopy(self.edit_poly)
            try:
                self.main.undo_stack.push(VertexEditCommand(self.main, self.active_actor, self.poly_before, poly_after))
            except Exception:
                pass

        # Rebuild transform gizmo if edit accepted
        if not cancel and self.active_actor:
            self.main.setup_transform_widget(self.active_actor)

        self.vtk_app.render_all()

    def _world_units_per_pixel_at_point(self, world_pt):
        cam = self.renderer.GetActiveCamera()
        # Perspective
        if cam.GetParallelProjection() == 0:
            import math
            cx, cy, cz = cam.GetPosition()
            dx = world_pt[0] - cx
            dy = world_pt[1] - cy
            dz = world_pt[2] - cz
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            fov_rad = math.radians(cam.GetViewAngle())
            # Height at distance
            view_height = 2.0 * dist * math.tan(fov_rad * 0.5)
            vp = self.renderer.GetViewport()
            win_w, win_h = self.window.GetSize()
            pix_h = max(int((vp[3]-vp[1]) * win_h), 1)
            return view_height / pix_h
        else:
            # Parallel projection
            vp = self.renderer.GetViewport()
            win_w, win_h = self.window.GetSize()
            pix_h = max(int((vp[3]-vp[1]) * win_h), 1)
            return (cam.GetParallelScale() * 2.0) / pix_h

    def _rescale_highlight(self):
        if not self.highlight_actor or not self.highlight_actor.GetVisibility():
            return
        pos = self.highlight_actor.GetPosition()
        units_per_px = self._world_units_per_pixel_at_point(pos)
        r = units_per_px * self._highlight_pixels
        r = max(self._min_world_radius, min(self._max_world_radius, r))
        self.highlight_source.SetRadius(r)
        self.highlight_source.Update()

    def _add_obs(self, event_name, cb):
        tag = self.iren.AddObserver(event_name, cb, 1.0)
        self.obs.append((event_name, tag))

    # ----- math helpers -----
    
    def _display_to_world(self, x, y, z_ndc):
        self.renderer.SetDisplayPoint(float(x), float(y), float(z_ndc))
        self.renderer.DisplayToWorld()
        wp = self.renderer.GetWorldPoint()
        if wp[3] == 0.0:
            return (0.0, 0.0, 0.0)
        return (wp[0] / wp[3], wp[1] / wp[3], wp[2] / wp[3])

    def _local_to_world(self, xyz):
        x, y, z = xyz
        m = self.actor_matrix
        vx = m.GetElement(0, 0) * x + m.GetElement(0, 1) * y + m.GetElement(0, 2) * z + m.GetElement(0, 3)
        vy = m.GetElement(1, 0) * x + m.GetElement(1, 1) * y + m.GetElement(1, 2) * z + m.GetElement(1, 3)
        vz = m.GetElement(2, 0) * x + m.GetElement(2, 1) * y + m.GetElement(2, 2) * z + m.GetElement(2, 3)
        return (vx, vy, vz)

    def _world_vec_to_local(self, dx, dy, dz):
        # Multiply by inverse matrix, w=0 (pure direction)
        m = self.actor_matrix_inv
        lx = m.GetElement(0, 0) * dx + m.GetElement(0, 1) * dy + m.GetElement(0, 2) * dz
        ly = m.GetElement(1, 0) * dx + m.GetElement(1, 1) * dy + m.GetElement(1, 2) * dz
        lz = m.GetElement(2, 0) * dx + m.GetElement(2, 1) * dy + m.GetElement(2, 2) * dz
        return (lx, ly, lz)

    def _intersect_drag_plane(self, sx, sy):
        # Ray from camera through screen point
        p0 = self._display_to_world(sx, sy, 0.0)
        p1 = self._display_to_world(sx, sy, 1.0)
        dx, dy, dz = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])

        nx, ny, nz = self.drag_plane_normal
        ox, oy, oz = self.drag_plane_origin

        denom = dx * nx + dy * ny + dz * nz
        if abs(denom) < 1e-9:
            return self.initial_plane_point_world

        t = ((ox - p0[0]) * nx + (oy - p0[1]) * ny + (oz - p0[2]) * nz) / denom
        return (p0[0] + dx * t, p0[1] + dy * t, p0[2] + dz * t)

    # ----- events -----
    def on_left_down(self, obj, evt):
        if not self.active_actor or not self.edit_poly:
            return
        x, y = self.iren.GetEventPosition()
        # Use picker ray to get a world point first
        self.picker.Pick(x, y, 0, self.renderer)
        pick_pos = self.picker.GetPickPosition()
        # Use locator for nearest vertex (stable even when zoomed)
        pid = self.point_locator.FindClosestPoint(pick_pos)
        if pid < 0:
            self.dragging = False
            return
        self.picked_pid = pid
        self.dragging = True
        # Plane origin at picked vertex world position
        pts = self.edit_poly.GetPoints()
        local = pts.GetPoint(pid)
        world = self._local_to_world(local)
        self.drag_plane_origin = world
        cam = self.renderer.GetActiveCamera()
        nx, ny, nz = cam.GetViewPlaneNormal()
        self.drag_plane_normal = (nx, ny, nz)
        self.initial_plane_point_world = world
        self.initial_local_pos = local
        if self.highlight_actor:
            self.highlight_actor.SetPosition(*world)
            self.highlight_actor.VisibilityOn()
            self._rescale_highlight()
        self.vtk_app.render_all()

    def on_left_up(self, obj, evt):
        # More robust: always terminate drag
        self.dragging = False
        self.picked_pid = -1
        self.initial_local_pos = (0.0, 0.0, 0.0)
        self.initial_plane_point_world = (0.0, 0.0, 0.0)
        if self.highlight_actor:
            self.highlight_actor.VisibilityOff()
        self.vtk_app.render_all()

    def on_mouse_move(self, obj, evt):
        # Abort drag if global left button no longer pressed (release missed)
        if self.dragging and not (QtWidgets.QApplication.mouseButtons() & QtCore.Qt.LeftButton):
            self.dragging = False
            self.picked_pid = -1
            if self.highlight_actor:
                self.highlight_actor.VisibilityOff()
            self.vtk_app.render_all()
            return

        if not self.dragging or self.picked_pid < 0 or not self.edit_poly:
            return

        x, y = self.iren.GetEventPosition()
        plane_pt = self._intersect_drag_plane(x, y)
        dx = plane_pt[0] - self.initial_plane_point_world[0]
        dy = plane_pt[1] - self.initial_plane_point_world[1]
        dz = plane_pt[2] - self.initial_plane_point_world[2]
        ldx, ldy, ldz = self._world_vec_to_local(dx, dy, dz)

        pts = self.edit_poly.GetPoints()
        px0, py0, pz0 = self.initial_local_pos
        new_pos = (px0 + ldx, py0 + ldy, pz0 + ldz)
        pts.SetPoint(self.picked_pid, *new_pos)
        pts.Modified()
        self.edit_poly.Modified()

        if self.highlight_actor:
            world_pos = self._local_to_world(new_pos)
            self.highlight_actor.SetPosition(*world_pos)
            self._rescale_highlight()

        self.main.update_scene_totals()
        self.vtk_app.render_all()

# ======= Undo/Redo Commands =======

class AddActorCommand(QUndoCommand):
    def __init__(self, main: MainWindow, base_name: str, actor: vtk.vtkActor):
        super().__init__(f"Add {base_name}")
        self.main = main
        self.base_name = base_name
        self.actor = actor
        self.unique_name = None

    def redo(self):
        if self.unique_name is None:
            self.unique_name = self.main._add_actor_no_undo(self.base_name, self.actor)
        else:
            self.main._add_actor_no_undo_with_name(self.unique_name, self.actor)

    def undo(self):
        if self.unique_name:
            self.main._remove_object_silent(self.unique_name)


class DeleteActorCommand(QUndoCommand):
    def __init__(self, main: MainWindow, unique_name: str, actor: vtk.vtkActor):
        super().__init__(f"Delete {unique_name}")
        self.main = main
        self.unique_name = unique_name
        self.actor = actor

    def redo(self):
        self.main._remove_object_silent(self.unique_name)

    def undo(self):
        self.main._add_actor_no_undo_with_name(self.unique_name, self.actor)


class TransformActorCommand(QUndoCommand):
    def __init__(self, main: MainWindow, actor: vtk.vtkActor, before_m16, after_m16):
        super().__init__("Transform")
        self.main = main
        self.actor = actor
        self.before = list(before_m16)
        self.after = list(after_m16)

    def redo(self):
        self.main._apply_user_matrix16(self.actor, self.after)

    def undo(self):
        self.main._apply_user_matrix16(self.actor, self.before)

class BatchPropertyChangeCommand(QUndoCommand):
    """Undo/redo for property changes across multiple actors."""
    def __init__(self, main: MainWindow, before_list, after_list):
        super().__init__(f"Change {len(before_list)} Object(s) Color")
        self.main = main
        # before_list / after_list: [(actor, snapshot_dict), ...]
        self.before = list(before_list)
        self.after = list(after_list)

    def redo(self):
        for actor, snap in self.after:
            self.main._apply_actor_property_snapshot(actor, snap)
        self.main.vtk_app.render_all()

    def undo(self):
        for actor, snap in self.before:
            self.main._apply_actor_property_snapshot(actor, snap)
        self.main.vtk_app.render_all()

class VertexEditCommand(QUndoCommand):
    def __init__(self, main: MainWindow, actor: vtk.vtkActor, poly_before: vtk.vtkPolyData, poly_after: vtk.vtkPolyData):
        super().__init__("Edit Vertices")
        self.main = main
        self.actor = actor

        self.before = vtk.vtkPolyData()
        self.before.DeepCopy(poly_before)
        self.after = vtk.vtkPolyData()
        self.after.DeepCopy(poly_after)

    def _apply_poly(self, poly):
        tp = vtk.vtkTrivialProducer()
        tp.SetOutput(poly)
        mapper = self.main.vtk_app.create_mapper(tp)
        self.actor.SetMapper(mapper)
        self.main.update_scene_totals()
        self.main.vtk_app.render_all()

    def redo(self):
        self._apply_poly(self.after)

    def undo(self):
        self._apply_poly(self.before)

class PropertyChangeCommand(QUndoCommand):
    def __init__(self, main: MainWindow, actor: vtk.vtkActor, before_snap: dict, after_snap: dict):
        super().__init__("Change Appearance")
        self.main = main
        self.actor = actor
        self.before = dict(before_snap)
        self.after = dict(after_snap)

    def redo(self):
        self.main._apply_actor_property_snapshot(self.actor, self.after)
        self.main.vtk_app.render_all()

    def undo(self):
        self.main._apply_actor_property_snapshot(self.actor, self.before)
        self.main.vtk_app.render_all()

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()