import inspect, sys
import semisim.physics.poisson as newp
print("NEW  poisson:", inspect.getfile(newp))

try:
    import physics.poisson as oldp
    print("OLD  poisson:", inspect.getfile(oldp))
except Exception as e:
    print("No top-level physics.poisson found:", e)

print("\nsys.path (first 5):")
print("\n".join(sys.path[:5]))
