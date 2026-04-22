from core.yaml_parser import parse_pipeline

def main() -> None:
    config = parse_pipeline("examples/titanic.yml")
    print(f"Pipeline name: {config.pipeline_name}")
    print("Stages:")
    for s in config.stages:
        print(f"  - {s.name!r} ({s.type})")
        if s.params:
            print(f"      params: {s.params}")
        if s.models:
            print(f"      models: {s.models}")

if __name__ == "__main__":
    main()
