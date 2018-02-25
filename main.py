import model
import analysis


def main():

    r = model.run(
        t_max=100, map_height=10, map_width=10,
        alpha=0.4, tau=0.01, movement_area=3, vision_area=5,
        x0=10, x1=10, x2=10, stride=1)
    analysis.separate.run(r)
    r.save()


if __name__ == "__main__":

    main()
